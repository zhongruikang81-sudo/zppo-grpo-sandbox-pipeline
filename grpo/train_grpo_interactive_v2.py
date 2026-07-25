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
        # input_ids shape: (batch_size, sequence_length)
        # scores shape: (batch_size, vocab_size)
        for i in range(input_ids.shape[0]):
            seq = input_ids[i]
            # Decode the last 8 tokens to check the actual string suffix
            lookback = min(8, len(seq))
            last_tokens = seq[-lookback:].tolist()
            last_text = self.tokenizer.decode(last_tokens)
            
            # If the generated text so far ends with a newline and optional whitespace,
            # we ban generating any more newline tokens to prevent consecutive newlines.
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
            
            # 1. Line-level repetition (3+ identical consecutive non-empty lines)
            lines = [l.strip() for l in last_text.split("\n") if l.strip()]
            if len(lines) >= 3:
                if lines[-1] == lines[-2] == lines[-3]:
                    scores[i, :] = float("-inf")
                    scores[i, self.eos_token_id] = 0.0
                    if isinstance(self.eot_token_id, int) and self.eot_token_id != self.tokenizer.unk_token_id:
                        scores[i, self.eot_token_id] = 0.0
                    continue
                    
            # 2. Punctuation/symbol repetition (15+ consecutive identical symbols)
            if len(last_text) >= 15:
                suffix = last_text[-15:]
                if len(set(suffix)) == 1 and suffix[0] in ".-_*+=#~":
                    scores[i, :] = float("-inf")
                    scores[i, self.eos_token_id] = 0.0
                    if isinstance(self.eot_token_id, int) and self.eot_token_id != self.tokenizer.unk_token_id:
                        scores[i, self.eot_token_id] = 0.0
                    continue
        return scores

# Define Stopping Criteria for </STEP> and consecutive newlines (fallback)
class StopOnStepEnd(StoppingCriteria):
    def __init__(self, stop_ids: List[int], tokenizer=None):
        self.stop_ids = list(stop_ids)
        self.n = len(self.stop_ids)
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0]
        # Check if </STEP> is generated
        if len(seq) >= self.n and list(seq[-self.n:].cpu().numpy()) == self.stop_ids:
            return True
        # Fallback check if 4 or more consecutive newlines are generated (should be blocked by logits processor)
        if self.tokenizer is not None and len(seq) >= 8:
            last_tokens = seq[-8:]
            last_text = self.tokenizer.decode(last_tokens)
            if last_text.endswith("\n\n\n\n"):
                return True
        return False

def get_inputs_and_labels(messages: List[Dict[str, str]], tokenizer) -> Tuple[List[int], List[int]]:
    """
    Encodes messages step-by-step to build input_ids and labels.
    Labels are set to -100 for all non-assistant tokens (including <start_of_turn>model\n).
    """
    input_ids = []
    labels = []
    
    current_messages = []
    for msg in messages:
        role = msg["role"]
        
        # Get prefix tokens
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
            # Mask out the "<start_of_turn>model\n" prefix (typically 3 tokens in Gemma-2: [106, 2516, 108])
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
    # NOTE: the original math_grpo_dataset_filtered.jsonl is not distributed;
    # the repo ships the equivalent numeric training mix under data/.
    dataset_path = str(REPO_ROOT / "data" / "numina_gsm_mix_numeric.jsonl")
    
    # Checkpoints structure for Stage 2
    start_step = 160
    latest_checkpoint = os.path.join(output_dir, "checkpoint_step160")
    
    # Auto-detect latest Stage 2 checkpoint to support resume
    if os.path.exists(output_dir):
        v2_checkpoints = []
        for d in os.listdir(output_dir):
            if d.startswith("checkpoint_step") and d.endswith("_v2") and os.path.isdir(os.path.join(output_dir, d)):
                try:
                    step_num = int(d.replace("checkpoint_step", "").replace("_v2", ""))
                    v2_checkpoints.append((step_num, os.path.join(output_dir, d)))
                except ValueError:
                    pass
        if v2_checkpoints:
            v2_checkpoints.sort()
            start_step, latest_checkpoint = v2_checkpoints[-1]
            print(f"Auto-detected Stage 2 checkpoint at step {start_step}: {latest_checkpoint}")
        else:
            print(f"No Stage 2 checkpoints found. Starting Stage 2 from baseline Checkpoint 160: {latest_checkpoint}")
            
        # Overwrite with DPO checkpoint if starting at or before step 230
        dpo_checkpoint = os.path.join(output_dir, "checkpoint_step230_v2_dpo")
        if os.path.exists(dpo_checkpoint) and start_step <= 230:
            print(f"Overriding checkpoint to DPO-aligned adapter for step 230: {dpo_checkpoint}")
            start_step = 230
            latest_checkpoint = dpo_checkpoint
            
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log_v2.txt")
    
    print("Loading tokenizer from model path...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Get stop token IDs for </STEP>
    stop_token_ids = tokenizer.encode("</STEP>", add_special_tokens=False)
    print(f"Stop token IDs for </STEP>: {stop_token_ids}")
    
    # Define EOT and EOS ids
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
    
    print("Initializing trainable PEFT adapter...")
    base_model = prepare_model_for_kbit_training(base_model)
    if os.path.exists(latest_checkpoint):
        print(f"Loading adapter weights from checkpoint: {latest_checkpoint}")
        model = PeftModel.from_pretrained(base_model, latest_checkpoint, is_trainable=True)
    else:
        # Fallback if checkpoint 160 directory is missing for some reason
        print(f"Warning: Checkpoint path {latest_checkpoint} not found. Starting from SFT base.")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["gate_proj", "v_proj", "k_proj", "q_proj", "down_proj", "o_proj", "up_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)
    model.warnings_issued = {}
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Optimizer and hyperparameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    
    # Restore optimizer state if checkpoint has one
    if os.path.exists(latest_checkpoint):
        opt_path = os.path.join(latest_checkpoint, "optimizer.pt")
        if os.path.exists(opt_path):
            print(f"Loading optimizer state from: {opt_path}")
            optimizer.load_state_dict(torch.load(opt_path, map_location="cuda"))
        else:
            print("Warning: No optimizer state found in checkpoint, starting optimizer fresh.")
    
    # Hyperparameters
    G = 4               # Group size
    max_turns = 3       # Max agent rollout turns
    clip_eps = 0.2      # PPO clip epsilon
    kl_coef = 0.01      # KL penalty coefficient
    max_steps = 2000      # Run up to 2000 steps total for Stage 2
    checkpoint_every = 20  # Save adapter checkpoint every N steps
    MAX_SEQ_LEN = 1664   # Hard cap on training sequence length to prevent OOM
    
    # Load GSM8K dataset if needed (for steps < 920)
    gsm8k_dataset = []
    if start_step < 920:
        print(f"Loading GSM8K dataset from {dataset_path}...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            gsm8k_dataset = [json.loads(line) for line in f]
        print(f"GSM8K loaded: {len(gsm8k_dataset)} samples")
        
    # Load Hendrycks MATH dataset if needed (for steps >= 920)
    math_by_level = {2: [], 3: [], 4: [], 5: []}
    if max_steps > 920:
        # NOTE: the original math_grpo_dataset_filtered.jsonl is not distributed;
        # the repo ships the equivalent numeric training mix under data/.
        math_path = str(REPO_ROOT / "data" / "hendrycks_math_grpo_numeric.jsonl")
        print(f"Loading MATH dataset from {math_path}...")
        if os.path.exists(math_path):
            with open(math_path, "r", encoding="utf-8") as f:
                math_dataset = [json.loads(line) for line in f]
            for item in math_dataset:
                lvl = item.get("level", 2)
                if lvl in math_by_level:
                    math_by_level[lvl].append(item)
            
            # Shuffle each level with a fixed seed for deterministic sampling
            import random
            for lvl in math_by_level:
                random.Random(42).shuffle(math_by_level[lvl])
                
            print(f"MATH loaded: {len(math_dataset)} samples")
            for lvl, items in math_by_level.items():
                print(f"  Level {lvl}: {len(items)} samples")
        else:
            print(f"Warning: MATH dataset not found at {math_path}!")
            
    # Start training loop
    log_mode = "a" if start_step > 160 else "w"
    with open(log_path, log_mode, encoding="utf-8") as log_file:
        if start_step == 160:
            log_file.write("Interactive GRPO Stage 2 Training Log\n======================================\n\n")
        
        for step in range(start_step, max_steps):
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            log_file.write(f"--- Step {step + 1} ---\n")
            
            # Draw one question from dataset based on current step
            if step < 920:
                sample = gsm8k_dataset[step % len(gsm8k_dataset)]
            else:
                import random
                
                def get_mix_ratios(s):
                    # Force 100% Level 3 training to stabilize formatting compliance
                    return {2: 0.00, 3: 1.00, 4: 0.00, 5: 0.00}
                        
                ratios = get_mix_ratios(step)
                levels = [2, 3, 4, 5]
                weights = [ratios[l] for l in levels]
                
                # Deterministic selection based on step number as seed
                step_rng = random.Random(step)
                selected_level = step_rng.choices(levels, weights=weights, k=1)[0]
                
                # Fetch question list for the selected level
                level_samples = math_by_level[selected_level]
                
                # Stateless deterministic sample selection
                sample = level_samples[(step * 13) % len(level_samples)]
                
            prompt = sample["prompt"]
            target_answer = sample.get("answer", None)
            
            # Inject supplementary prompt rules to steer model from sandbox hangs and NameErrors
            if "to solve the problem." in prompt:
                prompt = prompt.replace(
                    "to solve the problem.",
                    "to solve the problem. Note: Never use input() in Python code. Ensure all variables are defined before use in the current step.",
                    1
                )
            else:
                prompt = f"Never use input() in Python code. Ensure all variables are defined before use in the current step.\n{prompt}"
            
            print(f"Question: {prompt[:100]}...")
            log_file.write(f"Question: {prompt}\nTarget Answer: {target_answer}\n")
            
            # Rollout trajectories
            histories = []
            completions_text = []
            
            model.eval()
            # Enable KV Cache during generation rollout to restore normal inference speed
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
                        
                        # Extract newly generated tokens
                        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
                        assistant_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        
                        # Append assistant turn to history
                        history.append({"role": "assistant", "content": assistant_text})
                        
                        if "</STEP>" in assistant_text:
                            # Run accumulated code in sandbox
                            success, stdout, stderr = execute_accumulated_code(history)
                            observation = stdout.strip() if success else stderr.strip()
                            if not observation:
                                observation = "[Success: No output]" if success else "[Error: Unknown error]"
                            history.append({"role": "user", "content": f"Observation:\n{observation}\n"})
                        else:
                            # Assistant decided not to call code or finished
                            break
                            
                    histories.append(history)
                    
                    # Convert history to a single completion string for evaluation
                    comp_parts = []
                    for msg in history[1:]:
                        if msg["role"] == "assistant":
                            comp_parts.append(msg["content"])
                        else:
                            comp_parts.append(f"\nObservation:\n{msg['content']}\n")
                    completions_text.append("".join(comp_parts))
                    
            # Evaluate group rewards (Stage 2 version: passing prompt for C-OPRM context)
            rewards = calculate_group_rewards(completions_text, target_answer, prompt)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float, device="cuda")
            
            # Print outputs and rewards
            print(f"  Rewards: {rewards}")
            log_file.write(f"  Rewards: {rewards}\n")
            for idx, (comp, rew) in enumerate(zip(completions_text, rewards)):
                log_file.write(f"  --- Candidate {idx + 1} (Reward: {rew:.2f}) ---\n{comp}\n")
                
            # Calculate advantages
            std_r = torch.std(rewards_tensor)
            mean_r = torch.mean(rewards_tensor)
            if std_r < 1e-8:
                advantages = torch.zeros_like(rewards_tensor)
            else:
                advantages = (rewards_tensor - mean_r) / (std_r + 1e-8)
            # Disable KV Cache before training forward pass to prevent conflicts with gradient checkpointing
            model.config.use_cache = False
            
            # Empty CUDA cache to clear generation memory before backward passes
            torch.cuda.empty_cache()
            
            # Train step (accumulating gradients across candidates to save VRAM)
            model.train()
            optimizer.zero_grad()
            
            total_step_loss = 0.0
            for i in range(G):
                input_ids, labels = get_inputs_and_labels(histories[i], tokenizer)
                
                # Hard truncation to prevent OOM on extremely long multi-turn sequences
                if len(input_ids) > MAX_SEQ_LEN:
                    input_ids = input_ids[:MAX_SEQ_LEN]
                    labels = labels[:MAX_SEQ_LEN]
                
                input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")
                labels_tensor = torch.tensor([labels], dtype=torch.long, device="cuda")
                
                # Shift for causal LM loss calculation
                shift_labels = labels_tensor[0, 1:]
                shift_input_ids = input_ids_tensor[0, 1:]
                loss_mask = (shift_labels != -100)
                
                # Reference model forward pass (LoRA disabled)
                with torch.no_grad():
                    with model.disable_adapter():
                        ref_outputs = model(input_ids_tensor)
                        ref_logits = ref_outputs.logits
                        shift_ref_logits = ref_logits[0, :-1, :]
                        ref_log_probs = torch.log_softmax(shift_ref_logits, dim=-1)
                        per_token_ref_log_probs = ref_log_probs.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
                        per_token_ref_log_probs = per_token_ref_log_probs * loss_mask
                
                # Explicitly free ref model intermediate tensors before active forward pass
                del ref_outputs, ref_logits, shift_ref_logits, ref_log_probs
                torch.cuda.empty_cache()
                        
                # Forward pass with active LoRA adapter (retains activations for backward)
                outputs = model(input_ids_tensor)
                logits = outputs.logits
                shift_logits = logits[0, :-1, :]
                
                # Logprobs
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                per_token_log_probs = log_probs.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
                per_token_log_probs = per_token_log_probs * loss_mask
                
                # Mask out non-loss elements
                active_log_probs = per_token_log_probs[loss_mask]
                active_ref_log_probs = per_token_ref_log_probs[loss_mask]
                
                if len(active_log_probs) > 0:
                    ratios = torch.exp(active_log_probs - active_ref_log_probs)
                    adv = advantages[i]
                    
                    surr1 = ratios * adv
                    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                    clip_loss = -torch.min(surr1, surr2)
                    
                    kl = torch.exp(active_ref_log_probs - active_log_probs) - (active_ref_log_probs - active_log_probs) - 1.0
                    
                    total_token_loss = clip_loss + kl_coef * kl
                    n_tokens = len(active_log_probs)
                    loss = total_token_loss.sum() / (n_tokens * G)
                    
                    loss.backward()
                    total_step_loss += loss.item() * G
                    try:
                        del ratios, surr1, surr2, clip_loss, kl, total_token_loss, loss
                    except NameError:
                        pass
                else:
                    total_step_loss += 0.0
                
                # Free candidate tensors and clear cache between candidates
                del input_ids_tensor, labels_tensor
                try:
                    del outputs, logits, shift_logits, log_probs, per_token_log_probs, active_log_probs, active_ref_log_probs, per_token_ref_log_probs
                except NameError:
                    pass
                torch.cuda.empty_cache()
                    
            # Optimizer update step
            optimizer.step()
            
            print(f"  Loss: {total_step_loss:.4f}  |  Mean Reward: {mean_r.item():.4f}")
            log_file.write(f"  Loss: {total_step_loss:.4f}  |  Mean Reward: {mean_r.item():.4f}\n\n")
            log_file.flush()
            
            # Save checkpoint every checkpoint_every steps or at the final step (stage 2 naming suffix)
            if (step + 1) % checkpoint_every == 0 or (step + 1) == max_steps:
                ckpt_dir = os.path.join(output_dir, f"checkpoint_step{step + 1}_v2")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                # Save optimizer state for seamless resume
                torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                print(f"  [Checkpoint] Saved Stage 2 at step {step + 1} -> {ckpt_dir}")
                log_file.write(f"  [Checkpoint] Saved Stage 2 at step {step + 1}\n")
            
        print(f"\nSaving fine-tuned Stage 2 adapter to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Model saved successfully.")
        log_file.write("Stage 2 Training completed successfully.\n")

if __name__ == "__main__":
    main()
