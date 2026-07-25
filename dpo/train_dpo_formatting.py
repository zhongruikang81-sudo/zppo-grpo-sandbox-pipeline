import os
import sys
import torch
import json
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# Set HF mirror, disable tokenizer warnings, and prevent CUDA memory fragmentation
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Resolve repo root from this file's location (this script lives in <repo>/dpo/).
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent

def prepare_dpo_dataset(data_path, tokenizer):
    """
    Reads JSONL DPO pairs and formats them.
    Wraps the user prompt using Gemma-2's Chat Template, and appends EOS token.
    """
    print(f"Loading DPO pairs from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    formatted_data = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }
    
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        
        # 1. Format User prompt using Chat Template
        messages = [{"role": "user", "content": item["prompt"]}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 2. Add chosen and rejected completions with EOS token
        chosen_text = item["chosen"] + tokenizer.eos_token
        rejected_text = item["rejected"] + tokenizer.eos_token
        
        formatted_data["prompt"].append(formatted_prompt)
        formatted_data["chosen"].append(chosen_text)
        formatted_data["rejected"].append(rejected_text)
        
    print(f"Loaded {len(formatted_data['prompt'])} DPO pairs.")
    return Dataset.from_dict(formatted_data)

def main():
    parser = argparse.ArgumentParser(description="Run DPO alignment on checkpoint 1680 to enforce code formatting and correct library usage")
    # NOTE: the merged SFT base model (sft_merged, ~5GB) is NOT distributed with
    # this repository; pass --model-id pointing at your local merged model.
    parser.add_argument("--model-id", default=str(REPO_ROOT / "sft_merged"), help="Path to base model")
    parser.add_argument("--checkpoint-path", default=str(REPO_ROOT / "grpo_output_discounted" / "checkpoint_step1680_v2"), 
                        help="Path to the checkpoint adapter to fine-tune")
    parser.add_argument("--dpo-data-path", default=str(REPO_ROOT / "data" / "dpo_formatting_alignment.jsonl"), 
                        help="Path to the generated DPO dataset JSONL")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "grpo_output_discounted" / "checkpoint_step1680_v2_dpo"), 
                        help="Path to save the DPO-aligned adapter")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate (default: 5e-7)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2)")
    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Prepare dataset
    dataset = prepare_dpo_dataset(args.dpo_data_path, tokenizer)

    # 4-bit quantization config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load base model in 4-bit
    print("Loading base model in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare model for k-bit training
    base_model = prepare_model_for_kbit_training(base_model)

    # Load checkpoint LoRA adapter weights (trainable)
    if os.path.exists(args.checkpoint_path):
        print(f"Loading adapter weights from: {args.checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, args.checkpoint_path, is_trainable=True)
    else:
        print(f"Error: Checkpoint path {args.checkpoint_path} not found!")
        sys.exit(1)

    model.gradient_checkpointing_enable()

    # DPO Config & Training Arguments
    # Note: Using batch_size=1, gradient_accumulation_steps=8, and max_length=1664
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=1,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        beta=0.1,
        max_length=1024,  # Reverted back to 1024 to prevent CUDA OOM on 8GB GPU
        max_prompt_length=512,
        precompute_ref_log_probs=True,
    )

    print("Monkeypatching warnings_issued to prevent trl/peft compatibility issues...")
    model.warnings_issued = {}
    base_model.warnings_issued = {}
    
    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    # Save the aligned LoRA adapter and tokenizer
    print(f"Saving DPO-aligned adapter to: {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("DPO alignment completed successfully!")

if __name__ == "__main__":
    main()
