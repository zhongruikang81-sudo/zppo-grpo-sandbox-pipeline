import os
import sys
import re
import json
import torch
import textwrap
from typing import List, Dict, Tuple

# Set HF mirror, disable tokenizer warnings, and prevent CUDA memory fragmentation
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel, prepare_model_for_kbit_training

# Resolve repo root from this file's location so core/ is importable
# regardless of the current working directory.
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.sandbox import execute_accumulated_code
from core.evaluator import calculate_group_rewards

# Custom LogitsProcessor to suppress consecutive newlines
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

# Custom LogitsProcessor to suppress repetitive lines and character/symbol repetition
class SuppressRepetitiveContent(LogitsProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.eot_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for i in range(input_ids.shape[0]):
            seq = input_ids[i]
            lookback = min(150, len(seq))
            if lookback < 10:
                continue
            last_tokens = seq[-lookback:].tolist()
            last_text = self.tokenizer.decode(last_tokens)
            
            # Line-level repetition
            lines = [l.strip() for l in last_text.split("\n") if l.strip()]
            if len(lines) >= 3:
                if lines[-1] == lines[-2] == lines[-3]:
                    scores[i, :] = float("-inf")
                    scores[i, self.eos_token_id] = 0.0
                    if isinstance(self.eot_token_id, int) and self.eot_token_id != self.tokenizer.unk_token_id:
                        scores[i, self.eot_token_id] = 0.0
                    continue
                    
            # Punctuation/symbol repetition
            if len(last_text) >= 15:
                suffix = last_text[-15:]
                if len(set(suffix)) == 1 and suffix[0] in ".-_*+=#~":
                    scores[i, :] = float("-inf")
                    scores[i, self.eos_token_id] = 0.0
                    if isinstance(self.eot_token_id, int) and self.eot_token_id != self.tokenizer.unk_token_id:
                        scores[i, self.eot_token_id] = 0.0
                    continue
        return scores

# Stop criterion for </STEP>
class StopOnStepEnd(StoppingCriteria):
    def __init__(self, stop_ids: List[int], tokenizer=None):
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

def get_inputs_and_labels(messages: List[Dict[str, str]], tokenizer) -> Tuple[List[int], List[int]]:
    input_ids = []
    labels = []
    
    current_messages = []
    for msg in messages:
        role = msg["role"]
        
        if len(current_messages) == 0:
            prefix_tokens = []
        else:
            prefix_tokens = tokenizer.apply_chat_template(current_messages, tokenize=True, add_generation_prompt=False)
            if hasattr(prefix_tokens, "input_ids"):
                prefix_tokens = prefix_tokens.input_ids
            elif isinstance(prefix_tokens, dict) or hasattr(prefix_tokens, "keys"):
                prefix_tokens = prefix_tokens["input_ids"]
            prefix_tokens = list(prefix_tokens)
            
        current_messages.append(msg)
        full_tokens = tokenizer.apply_chat_template(current_messages, tokenize=True, add_generation_prompt=False)
        if hasattr(full_tokens, "input_ids"):
            full_tokens = full_tokens.input_ids
        elif isinstance(full_tokens, dict) or hasattr(full_tokens, "keys"):
            full_tokens = full_tokens["input_ids"]
        full_tokens = list(full_tokens)
        
        new_tokens = full_tokens[len(prefix_tokens):]
        input_ids.extend(new_tokens)
        
        if role == "assistant":
            assistant_labels = list(new_tokens)
            if len(assistant_labels) >= 3 and assistant_labels[:3] == [106, 2516, 108]:
                labels.extend([-100] * 3)
                labels.extend(assistant_labels[3:])
            else:
                labels.extend(assistant_labels)
        else:
            labels.extend([-100] * len(new_tokens))
            
    return input_ids, labels

def main():
    # NOTE: the merged SFT base model (sft_merged, ~5GB) is NOT distributed with
    # this repository. Point SFT_BASE_MODEL at your local merged model directory.
    model_id = os.environ.get("SFT_BASE_MODEL", str(REPO_ROOT / "sft_merged"))
    output_dir = str(REPO_ROOT / "grpo_output_discounted")
    dataset_path = str(REPO_ROOT / "data" / "numina_gsm_mix_numeric.jsonl")
    log_path = os.path.join(output_dir, "training_log_v3.txt")
    
    start_step = 1680
    latest_checkpoint = os.path.join(output_dir, "checkpoint_step1680_v2")
    
    # Auto-detect latest Stage 3 checkpoints
    if os.path.exists(output_dir):
        v3_checkpoints = []
        v2_checkpoints = []
        for d in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, d)):
                if d.startswith("checkpoint_step") and d.endswith("_v3"):
                    try:
                        step_num = int(d.replace("checkpoint_step", "").replace("_v3", ""))
                        v3_checkpoints.append((step_num, os.path.join(output_dir, d)))
                    except ValueError:
                        pass
                elif d.startswith("checkpoint_step") and d.endswith("_v2"):
                    try:
                        step_num = int(d.replace("checkpoint_step", "").replace("_v2", ""))
                        v2_checkpoints.append((step_num, os.path.join(output_dir, d)))
                    except ValueError:
                        pass
                        
        if v3_checkpoints:
            v3_checkpoints.sort()
            start_step, latest_checkpoint = v3_checkpoints[-1]
            print(f"Auto-detected Stage 3 checkpoint at step {start_step}: {latest_checkpoint}")
        elif v2_checkpoints:
            v2_checkpoints.sort()
            start_step, latest_checkpoint = v2_checkpoints[-1]
            print(f"Auto-detected Stage 2 checkpoint to resume from at step {start_step}: {latest_checkpoint}")
            
    print(f"Starting training from Step {start_step} using checkpoint: {latest_checkpoint}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    stop_token_ids = tokenizer.encode("</STEP>", add_special_tokens=False)
    eot_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [tokenizer.eos_token_id]
    if isinstance(eot_token_id, int) and eot_token_id != tokenizer.unk_token_id:
        eos_ids.append(eot_token_id)
        
    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Loading adapter...")
    base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(base_model, latest_checkpoint, is_trainable=True)
    model.warnings_issued = {}
    model.gradient_checkpointing_enable()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    opt_path = os.path.join(latest_checkpoint, "optimizer.pt")
    if os.path.exists(opt_path):
        print(f"Loading optimizer state from: {opt_path}")
        try:
            optimizer.load_state_dict(torch.load(opt_path, map_location="cuda"))
        except Exception as e:
            print(f"Warning: Failed to load optimizer state: {e}. Starting fresh.")
            
    # Load mixed numeric dataset
    print(f"Loading mixed training dataset: {dataset_path}")
    mixed_dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            mixed_dataset.append(json.loads(line))
    print(f"Loaded {len(mixed_dataset)} training samples.")
    
    # Hyperparameters
    G = 4
    max_turns = 3
    clip_eps = 0.2
    kl_coef = 0.01
    max_steps = 1800  # Train exactly up to step 1800
    checkpoint_every = 20
    MAX_SEQ_LEN = 1664
    
    log_mode = "a" if os.path.exists(log_path) else "w"
    with open(log_path, log_mode, encoding="utf-8") as log_file:
        if log_mode == "w":
            log_file.write("Interactive GRPO Stage 3 Training Log (Mixed Numeric Curriculum)\n======================================\n\n")
            
        for step in range(start_step, max_steps):
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            log_file.write(f"--- Step {step + 1} ---\n")
            
            # Select question from mixed dataset
            sample = mixed_dataset[step % len(mixed_dataset)]
            prompt = sample["prompt"]
            target_answer = sample["answer"]
            
            # Formatting prompt constraints
            if "You are a mathematical assistant." in prompt:
                if "to solve the problem." in prompt:
                    prompt = prompt.replace(
                        "to solve the problem.",
                        "to solve the problem. Note: Never use input() in Python code. Ensure all variables are defined before use in the current step.",
                        1
                    )
                else:
                    prompt = f"Note: Never use input() in Python code. Ensure all variables are defined before use in the current step.\n{prompt}"
            else:
                prompt = (
                    "You are a mathematical assistant. Use Python code wrapped inside <STEP>\n"
                    "```python\n"
                    "# code\n"
                    "```\n"
                    "</STEP> to solve the problem. Note: Never use input() in Python code. Ensure all variables are defined before use in the current step.\n\n"
                    f"{prompt}"
                )
                
            print(f"Question: {prompt[:100]}...")
            log_file.write(f"Question: {prompt}\nTarget Answer: {target_answer}\n")
            
            histories = []
            completions_text = []
            
            model.eval()
            model.config.use_cache = True
            with torch.no_grad():
                for idx in range(G):
                    print(f"  Generating candidate {idx + 1}/{G}...")
                    history = [{"role": "user", "content": prompt}]
                    
                    for turn in range(max_turns):
                        prompt_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                        inputs = tokenizer(prompt_str, return_tensors="pt").to("cuda")
                        
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            repetition_penalty=1.2,
                            eos_token_id=eos_ids,
                            stopping_criteria=StoppingCriteriaList([StopOnStepEnd(stop_token_ids, tokenizer)]),
                            logits_processor=LogitsProcessorList([
                                SuppressConsecutiveNewlines(tokenizer),
                                SuppressRepetitiveContent(tokenizer)
                            ])
                        )
                        
                        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
                        assistant_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        history.append({"role": "assistant", "content": assistant_text})
                        
                        if "</STEP>" in assistant_text:
                            success, stdout, stderr = execute_accumulated_code(history)
                            observation = stdout.strip() if success else stderr.strip()
                            if not observation:
                                observation = "[Success: No output]" if success else "[Error: Unknown error]"
                            history.append({"role": "user", "content": f"Observation:\n{observation}\n"})
                        else:
                            break
                            
                    histories.append(history)
                    
                    comp_parts = []
                    for msg in history[1:]:
                        if msg["role"] == "assistant":
                            comp_parts.append(msg["content"])
                        else:
                            comp_parts.append(f"\nObservation:\n{msg['content']}\n")
                    completions_text.append("".join(comp_parts))
                    
            # Calculate rewards (first-step penalty & self-correction bonus handled by evaluator_v2)
            rewards = calculate_group_rewards(completions_text, target_answer, prompt)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float, device="cuda")
            
            print(f"  Rewards: {rewards}")
            log_file.write(f"  Rewards: {rewards}\n")
            for idx, (comp, rew) in enumerate(zip(completions_text, rewards)):
                log_file.write(f"  --- Candidate {idx + 1} (Reward: {rew:.2f}) ---\n{comp}\n")
                
            std_r = torch.std(rewards_tensor)
            mean_r = torch.mean(rewards_tensor)
            if std_r < 1e-8:
                advantages = torch.zeros_like(rewards_tensor)
            else:
                advantages = (rewards_tensor - mean_r) / (std_r + 1e-8)
                
            model.config.use_cache = False
            torch.cuda.empty_cache()
            
            # Backpropagation
            model.train()
            optimizer.zero_grad()
            total_step_loss = 0.0
            
            for i in range(G):
                input_ids, labels = get_inputs_and_labels(histories[i], tokenizer)
                if len(input_ids) > MAX_SEQ_LEN:
                    input_ids = input_ids[:MAX_SEQ_LEN]
                    labels = labels[:MAX_SEQ_LEN]
                    
                input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")
                labels_tensor = torch.tensor([labels], dtype=torch.long, device="cuda")
                
                shift_labels = labels_tensor[0, 1:]
                shift_input_ids = input_ids_tensor[0, 1:]
                loss_mask = (shift_labels != -100)
                
                # Rollout-time (old) policy logprobs: adapter ENABLED, weights unchanged
                # since rollout (optimizer.step happens after the G loop), so this is
                # the true pi_theta_old required by the PPO/GRPO importance ratio.
                with torch.no_grad():
                    old_outputs = model(input_ids_tensor)
                    old_logits = old_outputs.logits
                    shift_old_logits = old_logits[0, :-1, :]
                    old_log_probs = torch.log_softmax(shift_old_logits, dim=-1)
                    per_token_old_log_probs = old_log_probs.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
                    per_token_old_log_probs = per_token_old_log_probs * loss_mask

                del old_outputs, old_logits, shift_old_logits, old_log_probs
                torch.cuda.empty_cache()

                # Reference logprobs (SFT base, adapter disabled) — used ONLY for the
                # KL-to-reference regularizer, NOT as the ratio denominator.
                with torch.no_grad():
                    with model.disable_adapter():
                        ref_outputs = model(input_ids_tensor)
                        ref_logits = ref_outputs.logits
                        shift_ref_logits = ref_logits[0, :-1, :]
                        ref_log_probs = torch.log_softmax(shift_ref_logits, dim=-1)
                        per_token_ref_log_probs = ref_log_probs.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
                        per_token_ref_log_probs = per_token_ref_log_probs * loss_mask
                        
                del ref_outputs, ref_logits, shift_ref_logits, ref_log_probs
                torch.cuda.empty_cache()
                
                # Active logprobs
                outputs = model(input_ids_tensor)
                logits = outputs.logits
                shift_logits = logits[0, :-1, :]
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                per_token_log_probs = log_probs.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
                per_token_log_probs = per_token_log_probs * loss_mask
                
                active_log_probs = per_token_log_probs[loss_mask]
                active_old_log_probs = per_token_old_log_probs[loss_mask]
                active_ref_log_probs = per_token_ref_log_probs[loss_mask]
                
                if len(active_log_probs) > 0:
                    # PPO/GRPO importance ratio: pi_theta / pi_theta_old (NOT vs SFT base)
                    ratios = torch.exp(active_log_probs - active_old_log_probs)
                    adv = advantages[i]
                    
                    surr1 = ratios * adv
                    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                    clip_loss = -torch.min(surr1, surr2)
                    
                    # KL-to-reference regularizer (k3 estimator, vs SFT base) — unchanged
                    kl = torch.exp(active_ref_log_probs - active_log_probs) - (active_ref_log_probs - active_log_probs) - 1.0
                    total_token_loss = clip_loss + kl_coef * kl
                    
                    loss = total_token_loss.sum() / (len(active_log_probs) * G)
                    loss.backward()
                    total_step_loss += loss.item() * G
                    
                    del ratios, surr1, surr2, clip_loss, kl, total_token_loss, loss
                else:
                    total_step_loss += 0.0
                    
                del input_ids_tensor, labels_tensor
                del outputs, logits, shift_logits, log_probs, per_token_log_probs, active_log_probs, active_old_log_probs, active_ref_log_probs, per_token_old_log_probs, per_token_ref_log_probs
                torch.cuda.empty_cache()
                
            optimizer.step()
            print(f"  Loss: {total_step_loss:.4f}  |  Mean Reward: {mean_r.item():.4f}")
            log_file.write(f"  Loss: {total_step_loss:.4f}  |  Mean Reward: {mean_r.item():.4f}\n\n")
            log_file.flush()
            
            # Save checkpoint
            if (step + 1) % checkpoint_every == 0 or (step + 1) == max_steps:
                ckpt_dir = os.path.join(output_dir, f"checkpoint_step{step + 1}_v3")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                print(f"  [Checkpoint] Saved Stage 3 at step {step + 1} -> {ckpt_dir}")
                log_file.write(f"  [Checkpoint] Saved Stage 3 at step {step + 1}\n")
                
        print("Saving final Stage 3 adapter...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Model saved successfully.")
        log_file.write("Stage 3 Training completed successfully.\n")

if __name__ == "__main__":
    main()
