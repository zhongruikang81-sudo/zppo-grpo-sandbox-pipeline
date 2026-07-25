import os
# 设置 Hugging Face 国内镜像，防止下载基础模型时连接超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Resolve repo root from this file's location (this script lives in <repo>/sft/).
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

def get_inputs_and_labels(messages, tokenizer):
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
            # Mask out "<start_of_turn>model\n" prefix (typically 3 tokens in Gemma-2: [106, 2516, 108])
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
    # 1. 定义配置参数
    # NOTE: 基座模型不随仓库分发。原始训练使用本地 gemma-2-2b-it 权重；
    # 通过 BASE_MODEL_ID 环境变量指向本地副本，默认从 HuggingFace 拉取。
    model_id = os.environ.get("BASE_MODEL_ID", "google/gemma-2-2b-it")
    # NOTE: math_sft_multiturn.jsonl 由 sft/generate_react_dataset.py 生成，不随仓库分发。
    dataset_path = str(REPO_ROOT / "data" / "math_sft_multiturn.jsonl")
    output_dir = str(REPO_ROOT / "sft_output")
    
    # 2. 加载 tokenizer（为了预先对数据进行 Tokenize 编码）
    print(f"Loading tokenizer: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 3. 加载数据集并进行预处理（合并 system prompt）
    print(f"Loading local multi-turn SFT dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    def preprocess_conversation(example):
        messages = example['messages']
        new_messages = []
        system_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                if system_content:
                    new_messages.append({"role": "user", "content": f"{system_content}\n\n{msg['content']}"})
                    system_content = ""
                else:
                    new_messages.append(msg)
            elif msg["role"] in ["assistant", "model"]:
                new_messages.append({"role": "assistant", "content": msg["content"]})
        return {"messages": new_messages}
        
    print("Preprocessing dataset to merge system prompts for Gemma-2...")
    dataset = dataset.map(preprocess_conversation, batched=False)

    # 4. 手动 Tokenize 整个数据集，并生成精确的 labels (排除了 user/observation 对应的 loss)
    def tokenize_function(example):
        ids, lbls = get_inputs_and_labels(example["messages"], tokenizer)
        return {
            "input_ids": ids,
            "labels": lbls
        }
        
    print("Tokenizing dataset and applying custom assistant-only loss masks...")
    dataset = dataset.map(tokenize_function, batched=False)
    # 仅保留训练所需的列
    dataset = dataset.select_columns(["input_ids", "labels"])

    # 5. 配置 4-bit 量化 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # 6. 加载 Model
    print(f"Loading base model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 7. 准备模型进行量化微调
    model = prepare_model_for_kbit_training(model)
    
    # 8. 配置 LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 9. 配置训练参数 (因为我们已经完成了 Tokenize 和 Label Masking，这里直接指定最简 SFTConfig)
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,       # 等效 batch_size = 16
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=2.0,                # 训练 2 个 Epoch
        optim="paged_adamw_32bit",
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        max_length=1024,
        packing=False
    )
    
    # 10. 初始化 TRL 训练器
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args
    )
    
    # 打印可训练参数量
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
        
    # 11. 开始 SFT 训练
    print("Starting pre-tokenized SFT training...")
    trainer.train()
    
    # 12. 保存微调权重
    print(f"Saving fine-tuned LoRA adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("SFT complete! Adapter saved successfully.")

if __name__ == "__main__":
    main()
