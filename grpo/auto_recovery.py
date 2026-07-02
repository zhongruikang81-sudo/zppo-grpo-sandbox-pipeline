import os
import sys
import re
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

# Reconfigure stdout for utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set HF mirror and env variables
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add math workspace to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_grpo_interactive_v2 import get_inputs_and_labels

def parse_comp_to_history(prompt, comp_text):
    history = [{"role": "user", "content": prompt}]
    parts = comp_text.split("\nObservation:\nObservation:\n")
    history.append({"role": "assistant", "content": parts[0]})
    
    for part in parts[1:]:
        idx = part.find("\n\n")
        if idx != -1:
            obs = part[:idx]
            assistant_turn = part[idx+2:]
            history.append({"role": "user", "content": f"Observation:\n{obs}\n"})
            history.append({"role": "assistant", "content": assistant_turn})
        else:
            history.append({"role": "user", "content": f"Observation:\n{part}\n"})
            
    return history

def get_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return 160, os.path.join(output_dir, "checkpoint_step160")
    
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
        return v2_checkpoints[-1]
    else:
        return 160, os.path.join(output_dir, "checkpoint_step160")

def get_latest_completed_step_from_log(log_path):
    if not os.path.exists(log_path):
        return 0
        
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    step_blocks = []
    matches = list(re.finditer(r'--- Step (\d+) ---', text))
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        step_num = int(matches[i].group(1))
        step_blocks.append((step_num, text[start:end]))
        
    latest_completed = 0
    for step_num, block in step_blocks:
        # Check if the block contains Loss indicating the step finished backward update
        if "Loss:" in block:
            if step_num > latest_completed:
                latest_completed = step_num
                
    return latest_completed

def run_replay(start_replay_step, end_replay_step, latest_ckpt_path, output_dir, log_path):
    print(f"\n=========================================")
    print(f"Starting Offline Replay Recovery Training")
    print(f"Replaying Steps: {start_replay_step} to {end_replay_step}")
    print(f"Base Checkpoint: {latest_ckpt_path}")
    print(f"=========================================\n")
    
    model_id = "E:\\math workspace\\sft_merged"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
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
    model = PeftModel.from_pretrained(base_model, latest_ckpt_path, is_trainable=True)
    model.gradient_checkpointing_enable()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    opt_path = os.path.join(latest_ckpt_path, "optimizer.pt")
    if os.path.exists(opt_path):
        print(f"Loading optimizer state from: {opt_path}")
        optimizer.load_state_dict(torch.load(opt_path, map_location="cuda"))
        
    # Read log and extract steps to replay
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    step_blocks = []
    matches = list(re.finditer(r'--- Step (\d+) ---', text))
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        step_num = int(matches[i].group(1))
        step_blocks.append((step_num, text[start:end]))
        
    extracted_steps = {}
    for step_num, block in step_blocks:
        if start_replay_step <= step_num <= end_replay_step:
            if step_num not in extracted_steps:
                extracted_steps[step_num] = block
                
    print(f"Found {len(extracted_steps)} steps to replay: {sorted(extracted_steps.keys())}")
    
    G = 4
    clip_eps = 0.2
    kl_coef = 0.01
    MAX_SEQ_LEN = 1664
    
    for s in sorted(extracted_steps.keys()):
        block_text = extracted_steps[s]
        
        # Extract prompt
        q_match = re.search(r'Question:\s*(.*?)(?=\nTarget Answer:)', block_text, re.DOTALL)
        if not q_match:
            print(f"Warning: Could not parse question for step {s}. Skipping.")
            continue
        prompt = q_match.group(1).strip()
        
        # Extract target answer
        ta_match = re.search(r'Target Answer:\s*(.*?)(?=\n)', block_text)
        target_answer = ta_match.group(1).strip() if ta_match else ""
        
        # Extract rewards
        rewards_match = re.search(r'Rewards:\s*\[(.*?)\]', block_text)
        if not rewards_match:
            print(f"Warning: Could not parse rewards for step {s}. Skipping.")
            continue
        rewards = [float(r.strip()) for r in rewards_match.group(1).split(',')]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device="cuda")
        
        # Extract candidates
        candidates = re.split(r'--- Candidate \d+ \(Reward: .*?\) ---', block_text)
        if len(candidates) - 1 < G:
            print(f"Warning: Insufficient candidates ({len(candidates)-1}/{G}) for step {s}. Skipping.")
            continue
            
        histories = []
        for cand_body in candidates[1:G+1]:
            comp_text = cand_body.strip()
            # Clean up trailing Loss lines if it's the last candidate
            comp_text = re.split(r'\n\s*Loss:|\n\s*\[Checkpoint\]', comp_text)[0].strip()
            hist = parse_comp_to_history(prompt, comp_text)
            histories.append(hist)
            
        # Calculate advantages
        std_r = torch.std(rewards_tensor)
        mean_r = torch.mean(rewards_tensor)
        if std_r < 1e-8:
            advantages = torch.zeros_like(rewards_tensor)
        else:
            advantages = (rewards_tensor - mean_r) / (std_r + 1e-8)
            
        model.config.use_cache = False
        torch.cuda.empty_cache()
        
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
            
            # Reference model forward pass (LoRA disabled)
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
            
            # Active model forward pass (LoRA enabled)
            outputs = model(input_ids_tensor)
            logits = outputs.logits
            shift_logits = logits[0, :-1, :]
            
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            per_token_log_probs = log_probs.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)
            per_token_log_probs = per_token_log_probs * loss_mask
            
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
                
                del ratios, surr1, surr2, clip_loss, kl, total_token_loss, loss
            else:
                total_step_loss += 0.0
                
            del input_ids_tensor, labels_tensor
            try:
                del outputs, logits, shift_logits, log_probs, per_token_log_probs, active_log_probs, active_ref_log_probs, per_token_ref_log_probs
            except NameError:
                pass
            torch.cuda.empty_cache()
            
        optimizer.step()
        print(f"Replay Step {s} Completed. Loss: {total_step_loss:.4f} | Mean Reward: {mean_r.item():.4f}")
        
    # Save the replayed checkpoint
    ckpt_dir = os.path.join(output_dir, f"checkpoint_step{end_replay_step}_v2")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    print(f"\nReplay recovery finished successfully! Saved updated weights to: {ckpt_dir}")

def main():
    output_dir = r"E:\math workspace\grpo_output_discounted"
    log_path = os.path.join(output_dir, "training_log_v2.txt")
    
    latest_ckpt_step, latest_ckpt_path = get_latest_checkpoint(output_dir)
    latest_completed_step = get_latest_completed_step_from_log(log_path)
    
    print(f"Latest saved checkpoint step: {latest_ckpt_step} ({latest_ckpt_path})")
    print(f"Latest completed step in log: {latest_completed_step}")
    
    lost_steps = latest_completed_step - latest_ckpt_step
    print(f"Number of lost steps: {lost_steps}")
    
    if lost_steps > 10:
        print(f"Lost steps {lost_steps} > 10. Running replay equivalent training...")
        run_replay(latest_ckpt_step + 1, latest_completed_step, latest_ckpt_path, output_dir, log_path)
        print("Replay completed successfully.")
    elif lost_steps > 0:
        print(f"Lost steps {lost_steps} <= 10. No replay needed per user rules. Will resume directly.")
    else:
        print("No lost steps detected. Checkpoint matches the log.")

if __name__ == "__main__":
    main()
