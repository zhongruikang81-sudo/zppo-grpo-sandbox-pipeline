import os
import sys
import gc
import re
import json
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from typing import List, Dict, Any, Tuple

# Resolve repo root from this file's location; add both the repo root (for
# core/ imports) and this script's directory (for zppo_buffer).
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from core.sandbox import execute_accumulated_code
from core.evaluator import compare_math_answers, get_prm_salvage_score
from zppo_buffer import PromptReplayBuffer

# Definitions of custom LogitsProcessor and StoppingCriteria
class SuppressConsecutiveNewlines(LogitsProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.newline_tokens = {108, 109, 110}
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i in range(input_ids.shape[0]):
            seq = input_ids[i]
            lookback = min(8, len(seq))
            last_tokens = seq[-lookback:].tolist()
            last_text = self.tokenizer.decode(last_tokens)
            
            if re.search(r'\n\s*$', last_text):
                for tok_id in self.newline_tokens:
                    scores[i, tok_id] = float("-inf")
                    
            # Always ban compound newline tokens
            scores[i, 109] = float("-inf")
            scores[i, 110] = float("-inf")
        return scores

class SuppressRepetitiveContent(LogitsProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.eot_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i in range(input_ids.shape[0]):
            seq = input_ids[i]
            lookback = min(200, len(seq))
            if lookback < 10:
                continue
            last_tokens = seq[-lookback:].tolist()
            last_text = self.tokenizer.decode(last_tokens)
            
            lines = [l.strip() for l in last_text.split("\n") if l.strip()]
            
            # Multi-line loop detection (1 to 8 lines repeating 3+ times)
            has_loop = False
            for length in range(1, 9):
                if len(lines) >= length * 3:
                    # Look at the end of the line list
                    for start_idx in range(len(lines) - length * 3, len(lines) - length * 3 + 1):
                        pattern = lines[start_idx : start_idx + length]
                        match = True
                        for rep in range(1, 3):
                            test_pattern = lines[start_idx + length * rep : start_idx + length * (rep + 1)]
                            if pattern != test_pattern:
                                match = False
                                break
                        if match:
                            has_loop = True
                            break
                if has_loop:
                    break
            
            # Punctuation/symbol repetition (15 consecutive same chars)
            has_punc_rep = False
            if len(last_text) >= 15:
                suffix = last_text[-15:]
                if len(set(suffix)) == 1 and suffix[0] in ".-_*+=#~":
                    has_punc_rep = True
                    
            if has_loop or has_punc_rep:
                scores[i, :] = float("-inf")
                scores[i, self.eos_token_id] = 0.0
                if isinstance(self.eot_token_id, int) and self.eot_token_id != self.tokenizer.unk_token_id:
                    scores[i, self.eot_token_id] = 0.0
        return scores

class StopOnStepEnd(StoppingCriteria):
    def __init__(self, stop_ids, tokenizer=None):
        self.stop_ids = list(stop_ids)
        self.n = len(self.stop_ids)
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0]
        if len(seq) >= self.n and list(seq[-self.n:].cpu().numpy()) == self.stop_ids:
            return True
        if self.tokenizer is not None and len(seq) >= 8:
            last_tokens = seq[-8:]
            last_text = self.tokenizer.decode(last_tokens)
            if last_text.endswith("\n\n\n\n"):
                return True
        return False

# Training Settings
# NOTE: sft_merged (~5GB) is not distributed with this repo; set SFT_BASE_MODEL.
MODEL_ID = os.environ.get("SFT_BASE_MODEL", str(REPO_ROOT / "sft_merged"))
DATASET_PATH = str(REPO_ROOT / "data" / "numina_gsm_mix_numeric.jsonl")
OUTPUT_DIR = str(REPO_ROOT / "grpo_output_discounted" / "zppo_checkpoint")
PRB_STATE_PATH = str(REPO_ROOT / "grpo_output_discounted" / "zppo_prb_state.json")

BUFFER_SIZE = 8          # Mini-buffer size for local GPU training
GRAD_THRESHOLD = 2       # Graduate question after 2 consecutive correct rollouts
MAX_STEPS = 5            # Retire question after 5 steps in buffer
TRAINING_STEPS = 1000    # Total optimization steps (approx. 12-13 hours of training)
LEARNING_RATE = 5e-6
BETA = 0.01              # KL penalty coefficient
EPSILON = 0.2            # PPO clip parameter

def get_reward_for_student_response(question: str, target: str, response_text: str, history: List[Dict[str, str]]) -> Tuple[float, bool]:
    """
    Grades the student rollout response using local math checker + DeepSeek judge.
    Returns (reward, is_correct)
    """
    # 1. Run local equivalence check first (to save API cost)
    local_matched = compare_math_answers(response_text, target)
    if local_matched:
        return 1.0, True
        
    # 2. Call DeepSeek API judge if local check fails
    print(f"    [Local Check Failed] Invoking DeepSeek PRM Judge for Q: {question[:60]}...")
    score = get_prm_salvage_score(response_text, target, question)
    print(f"    PRM Judge Score: {score}")
    if score >= 0.50:
        return 1.0, True
    return score * 2.0, False
    
def get_student_trajectory(model, tokenizer, question: str, target_answer: str, max_turns: int = 3) -> Tuple[bool, str]:
    """
    Runs up to max_turns of student model interaction with the local sandbox.
    Returns (is_correct, trajectory_text)
    """
    prompt = (
        "You are a mathematical assistant. Use Python code wrapped inside <STEP>\n"
        "```python\n"
        "# code\n"
        "```\n"
        "</STEP> to solve the problem. Note: Never use input() in Python code. Ensure all variables are defined before use in the current step.\n\n"
        f"{question}"
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    eos_ids = [tokenizer.eos_token_id]
    eot_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot_token_id, int) and eot_token_id != tokenizer.unk_token_id:
        eos_ids.append(eot_token_id)
        
    stop_token_ids = tokenizer.encode("</STEP>", add_special_tokens=False)
    
    model.eval()
    
    for turn in range(max_turns):
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_str, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Greedy decoding for initial evaluation
                eos_token_id=eos_ids,
                stopping_criteria=StoppingCriteriaList([StopOnStepEnd(stop_token_ids, tokenizer)]),
                logits_processor=LogitsProcessorList([
                    SuppressConsecutiveNewlines(tokenizer),
                    SuppressRepetitiveContent(tokenizer)
                ])
            )
            
        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        assistant_content = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        messages.append({"role": "assistant", "content": assistant_content})
        
        has_code = "</STEP>" in assistant_content
        if has_code:
            success, stdout, stderr = execute_accumulated_code(messages)
            observation = stdout.strip() if success else stderr.strip()
            if not observation:
                observation = "[Success: No output]" if success else "[Error: Unknown error]"
            messages.append({"role": "user", "content": f"Observation:\n{observation}\n"})
        else:
            break
            
    # Compile final trajectory text
    final_parts = []
    for m in messages[1:]:
        if m["role"] == "assistant":
            final_parts.append(m["content"])
        else:
            final_parts.append(f"\n{m['content']}")
            
    student_trajectory = "".join(final_parts)
    
    # Check correctness
    is_correct = compare_math_answers(student_trajectory, target_answer)
    if not is_correct:
        score = get_prm_salvage_score(student_trajectory, target_answer, question)
        if score >= 0.50:
            is_correct = True
            
    return is_correct, student_trajectory

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.stdout.reconfigure(encoding='utf-8')
    
    # ------------------ Phase 1: Load Student Model & Tokenizer (Loaded First Now) ------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Extract stop token IDs and setup stopping/logits lists
    stop_token_ids = tokenizer.encode("</STEP>", add_special_tokens=False)
    eot_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [tokenizer.eos_token_id]
    if isinstance(eot_token_id, int) and eot_token_id != tokenizer.unk_token_id:
        eos_ids.append(eot_token_id)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    print("Loading base SFT model in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Determine the starting checkpoint (resume from latest ZPPO checkpoint if available)
    GRPO_CHECKPOINT_PATH = str(REPO_ROOT / "grpo_output_discounted" / "checkpoint_step1800_v3")
    starting_checkpoint = GRPO_CHECKPOINT_PATH
    
    if os.path.exists(OUTPUT_DIR):
        zppo_checkpoints = []
        for d in os.listdir(OUTPUT_DIR):
            if d.startswith("checkpoint_step") and os.path.isdir(os.path.join(OUTPUT_DIR, d)):
                try:
                    step_num = int(d.replace("checkpoint_step", ""))
                    zppo_checkpoints.append((step_num, os.path.join(OUTPUT_DIR, d)))
                except ValueError:
                    pass
        if zppo_checkpoints:
            zppo_checkpoints.sort()
            starting_checkpoint = zppo_checkpoints[-1][1]
            print(f"Detected existing ZPPO checkpoints. Resuming from latest: {starting_checkpoint} (step {zppo_checkpoints[-1][0]}).")
        else:
            print(f"No ZPPO checkpoints found in {OUTPUT_DIR}. Starting from base GRPO adapter: {GRPO_CHECKPOINT_PATH}")
    else:
        print(f"No ZPPO output directory found. Starting from base GRPO adapter: {GRPO_CHECKPOINT_PATH}")
        
    base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(base_model, starting_checkpoint, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # ------------------ Phase 2: Initialize Prompt Replay Buffer ------------------
    prb = PromptReplayBuffer(
        dataset_path=DATASET_PATH, 
        buffer_size=BUFFER_SIZE, 
        grad_threshold=GRAD_THRESHOLD, 
        max_steps=MAX_STEPS,
        start_index=0  # Start from 0 since we are training SFT from scratch on NuminaMath, keeping 1980+ for evaluation
    )
    
    def student_eval_fn(question: str, target_answer: str) -> Tuple[bool, str]:
        return get_student_trajectory(model, tokenizer, question, target_answer)
        
    # Load state if it exists
    if os.path.exists(PRB_STATE_PATH):
        prb.load_state(PRB_STATE_PATH)
    else:
        prb.replenish(student_eval_fn=student_eval_fn)
        prb.save_state(PRB_STATE_PATH)
        
    # Detailed log file path
    ZPPO_LOG_PATH = str(REPO_ROOT / "grpo_output_discounted" / "zppo_training_log.txt")
    
    # Append mode to avoid overwriting previous steps when resuming
    log_mode = "a" if os.path.exists(ZPPO_LOG_PATH) else "w"
    
    # ------------------ Phase 3: ZPPO Training Loop ------------------
    print("\n==============================================")
    print("Starting ZPPO Distillation Training Loop...")
    print("==============================================")
    
    with open(ZPPO_LOG_PATH, log_mode, encoding="utf-8") as log_file:
        if log_mode == "w":
            log_file.write("ZPPO Distillation Training Log\n============================================\n\n")
            
        start_step = prb.current_step
        for step in range(start_step, TRAINING_STEPS):
            print(f"\n--- [ZPPO Step {step + 1}/{TRAINING_STEPS}] ---")
            log_file.write(f"--- [ZPPO Step {step + 1}/{TRAINING_STEPS}] ---\n")
            
            # 1. Sample active buffer items for rollout
            if not prb.buffer:
                print("PRB is empty. Replenishing...")
                prb.replenish()
                prb.save_state(PRB_STATE_PATH)
                
            # Draw 1 question for GRPO single-prompt double-sampling (G=2)
            batch_items = random.sample(prb.buffer, 1)
            item = batch_items[0]
            
            step_losses = []
            step_kls = []
            step_rewards = []
            
            # ── Phase A: Collect ALL rollouts first (no backward yet) ──────────
            rollout_data = []  # list of dicts: {index, prompt_ids, gen_ids, reward, is_correct, clean_traj}
            
            index = item["index"]
            bcq_prompt = prb.get_bcq_prompt(item)
            target = item["target"]
            
            log_file.write(f"\n  [Item Index: {index}]\n")
            log_file.write(f"  Core Question:\n{item['core_question']}\n\n")
            log_file.write(f"  Target Answer: {target}\n\n")
            log_file.write(f"  Constructed BCQ Prompt:\n{bcq_prompt}\n\n")
            log_file.write(f"  Teacher Trajectory:\n{item['teacher_trajectory']}\n\n")
            
            # Format using tokenizer chat template (same for both rollouts)
            chat = [{"role": "user", "content": bcq_prompt}]
            prompt_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            
            is_correct_any = False
            clean_traj_any = None
            
            for g in range(2):
                log_file.write(f"  --- Rollout {g + 1}/2 ---\n")
                # Rollout generation (with exploration sampling)
                model.eval()
                inputs = tokenizer(prompt_str, return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        eos_token_id=eos_ids,
                        stopping_criteria=StoppingCriteriaList([StopOnStepEnd(stop_token_ids, tokenizer)]),
                        logits_processor=LogitsProcessorList([
                            SuppressConsecutiveNewlines(tokenizer),
                            SuppressRepetitiveContent(tokenizer)
                        ])
                    )
                    
                gen_ids = outputs[0][inputs.input_ids.shape[1]:]
                student_response = tokenizer.decode(gen_ids, skip_special_tokens=True)
                
                log_file.write(f"  Student Response {g + 1}:\n{student_response}\n\n")
                
                # Simulate sandbox execution for reward
                history = [{"role": "user", "content": bcq_prompt}, {"role": "assistant", "content": student_response}]
                has_code = "</STEP>" in student_response
                observation = ""
                
                if has_code:
                    success, stdout, stderr = execute_accumulated_code(history)
                    observation = stdout.strip() if success else stderr.strip()
                    if not observation:
                        observation = "[Success: No output]" if success else "[Error: Unknown error]"
                    history.append({"role": "user", "content": f"Observation:\n{observation}\n"})
                    log_file.write(f"  Sandbox Observation:\n{observation}\n\n")
                    
                    reward, is_correct = get_reward_for_student_response(item["core_question"], target, student_response, history)
                    
                    match = re.search(r'(<STEP>.*?</STEP>)', student_response, re.DOTALL)
                    if match:
                        code_block = match.group(1)
                        clean_student_traj = f"{code_block}\nObservation:\n{observation}\n"
                    else:
                        clean_student_traj = None
                else:
                    reward, is_correct = 0.0, False
                    log_file.write("  Sandbox Observation: [No code block generated]\n\n")
                    clean_student_traj = None
                    
                print(f"  Item index {index} | Rollout {g + 1}/2 | Reward: {reward} | Is Correct: {is_correct}")
                log_file.write(f"  Reward {g + 1}: {reward} | Is Correct: {is_correct}\n\n")
                step_rewards.append(reward)
                
                if is_correct:
                    is_correct_any = True
                    if clean_student_traj:
                        clean_traj_any = clean_student_traj
                
                # Compute old policy logprobs for ratio calculation (before weight update)
                response_ids = gen_ids.tolist()
                full_ids     = prompt_ids + response_ids
                input_tensor_old = torch.tensor([full_ids]).to("cuda")
                attention_mask_old = torch.ones_like(input_tensor_old)
                with torch.no_grad():
                    outputs_old    = model(input_ids=input_tensor_old, attention_mask=attention_mask_old)
                    logits_old     = outputs_old.logits
                    shift_logits_o = logits_old[..., :-1, :].contiguous()
                    shift_labels_o = input_tensor_old[..., 1:].contiguous()
                    log_probs_old  = torch.log_softmax(shift_logits_o, dim=-1)
                    token_lp_old   = log_probs_old.gather(dim=-1, index=shift_labels_o.unsqueeze(-1)).squeeze(-1).detach().cpu()
                
                del outputs_old, logits_old, shift_logits_o, log_probs_old, input_tensor_old, attention_mask_old
                
                rollout_data.append({
                    "index": index,
                    "prompt_ids": prompt_ids,
                    "gen_ids": gen_ids.cpu(),
                    "reward": reward,
                    "token_lp_old": token_lp_old,
                })
                
                del outputs, inputs
                gc.collect()
                torch.cuda.empty_cache()
                
            # Update PRB question stats once per step for this question
            prb.update_buffer_item(index, clean_traj_any, is_correct_any)
            
            # ── Phase B: Zero-Advantage Skip ──────────────────────────────────
            # If ALL rewards in the batch are identical, Advantage == 0 for every
            # rollout → gradients are exactly zero → skip backward entirely.
            unique_rewards = set(round(r["reward"], 6) for r in rollout_data)
            if len(unique_rewards) <= 1:
                skipped_msg = f"  [SKIP] All rollouts have reward={list(unique_rewards)}. Zero advantage — skipping backward."
                print(skipped_msg)
                log_file.write(skipped_msg + "\n")
            else:
                # ── Phase C: True batch-level advantage + combined backward ───
                avg_reward = sum(r["reward"] for r in rollout_data) / len(rollout_data)
                model.train()
                optimizer.zero_grad()   # single zero_grad for entire step
                
                accumulated_loss = 0.0
                accumulated_kl   = 0.0
                valid_rollouts   = 0
                
                for rd in rollout_data:
                    prompt_ids_rd = rd["prompt_ids"]
                    gen_ids_rd    = rd["gen_ids"]
                    reward_rd     = rd["reward"]
                    advantage     = reward_rd - avg_reward
                    
                    response_ids = gen_ids_rd.tolist()
                    full_ids     = prompt_ids_rd + response_ids
                    
                    input_tensor    = torch.tensor([full_ids]).to("cuda")
                    attention_mask  = torch.ones_like(input_tensor)
                    
                    # Forward: Student Policy (LoRA enabled)
                    outputs_theta   = model(input_ids=input_tensor, attention_mask=attention_mask)
                    logits_theta    = outputs_theta.logits
                    shift_logits_t  = logits_theta[..., :-1, :].contiguous()
                    shift_labels    = input_tensor[..., 1:].contiguous()
                    log_probs_theta = torch.log_softmax(shift_logits_t, dim=-1)
                    token_lp_theta  = log_probs_theta.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                    
                    del outputs_theta, logits_theta, shift_logits_t, log_probs_theta
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Forward: Reference Policy (LoRA disabled)
                    with torch.no_grad():
                        with model.disable_adapter():
                            outputs_ref   = model(input_ids=input_tensor, attention_mask=attention_mask)
                            logits_ref    = outputs_ref.logits
                            shift_logits_r = logits_ref[..., :-1, :].contiguous()
                            log_probs_ref  = torch.log_softmax(shift_logits_r, dim=-1)
                            token_lp_ref   = log_probs_ref.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                            
                            del outputs_ref, logits_ref, shift_logits_r, log_probs_ref
                            gc.collect()
                            torch.cuda.empty_cache()
                    
                    # Label mask (response tokens only, including final EOS to learn stopping)
                    prompt_len = len(prompt_ids_rd)
                    seq_len    = len(full_ids)
                    mask = torch.zeros(seq_len - 1, dtype=torch.float32).to("cuda")
                    mask[prompt_len - 1:] = 1.0
                    
                    num_tokens = mask.sum()
                    if num_tokens == 0:
                        print(f"  [Warning] Rollout {rd['index']}: no response tokens. Skipping.")
                        del token_lp_theta, token_lp_ref, input_tensor, attention_mask, mask
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    
                    # PPO surrogate + KL
                    token_lp_old      = rd["token_lp_old"].to("cuda")
                    token_ratio       = torch.exp(token_lp_theta - token_lp_old)
                    surrogate_1       = token_ratio * advantage
                    surrogate_2       = torch.clamp(token_ratio, 1 - EPSILON, 1 + EPSILON) * advantage
                    token_policy_loss = -torch.min(surrogate_1, surrogate_2)
                    token_kl          = token_lp_theta - token_lp_ref
                    total_token_loss  = token_policy_loss + BETA * token_kl
                    
                    masked_loss = (total_token_loss * mask).sum() / num_tokens
                    # Accumulate (will call backward after the loop for last rollout,
                    # or accumulate gradients across rollouts)
                    masked_loss.backward()
                    
                    step_losses.append(masked_loss.item())
                    step_kls.append((token_kl * mask).sum().item() / num_tokens.item())
                    accumulated_loss += masked_loss.item()
                    valid_rollouts   += 1
                    
                    del token_lp_theta, token_lp_ref, token_lp_old, token_ratio, token_policy_loss, token_kl
                    del total_token_loss, masked_loss, input_tensor, attention_mask, mask
                    gc.collect()
                    torch.cuda.empty_cache()
                
                if valid_rollouts > 0:
                    optimizer.step()
                else:
                    print("  [Warning] No valid rollouts in this step. Optimizer step skipped.")
            
            # ── Phase D: Clean Buffer and Save State ─────────────────────────
            graduated, retired = prb.clean_and_replenish(student_eval_fn=student_eval_fn)
            prb.current_step = step + 1
            prb.save_state(PRB_STATE_PATH)
            
            # Save checkpoints periodically
            if (step + 1) % 20 == 0:
                chk_dir = os.path.join(OUTPUT_DIR, f"checkpoint_step{step + 1}")
                model.save_pretrained(chk_dir)
                tokenizer.save_pretrained(chk_dir)
                print(f"  [Checkpoint] Saved step {step + 1} model to {chk_dir}")
                
            avg_loss = sum(step_losses) / len(step_losses) if step_losses else 0.0
            avg_kl   = sum(step_kls) / len(step_kls) if step_kls else 0.0
            avg_r    = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
            print(f"Step {step+1} Summary | Loss: {avg_loss:.4f} | KL: {avg_kl:.4f} | Avg Reward: {avg_r:.2f} | Buffer Graduated: {graduated} | Retired: {retired}")
            log_file.write(f"Step {step+1} Summary | Loss: {avg_loss:.4f} | KL: {avg_kl:.4f} | Avg Reward: {avg_r:.2f} | Buffer Graduated: {graduated} | Retired: {retired}\n\n")
            log_file.flush()
            
    # Save final weights
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nZPPO Distillation Complete. Final model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
