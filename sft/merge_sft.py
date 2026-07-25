"""
merge_sft.py
============
离线将 SFT LoRA adapter 合并进 base model，生成新的基座模型。

用途：
  GRPO 训练的 Reference Model 应该是 SFT 模型的冻结快照，而不是原始 base。
  通过把 SFT LoRA 合并进 base 的权重，再套一层全新的 GRPO LoRA，
  disable_adapter() 时 reveal 的就是正确的 SFT 基座，KL 初始值为 0。

运行方式：
  python merge_sft.py

注意：
  - 合并在 CPU + bfloat16 下进行，不需要显存，但需要约 6~8GB 系统内存。
  - 合并完成后，输出目录可作为 train_grpo_interactive.py 的新 model_id。
  - 合并完成后，train_grpo_interactive.py 的 adapter_id 改为 None，
    并添加 LoraConfig 从头初始化 GRPO LoRA。
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

# Repo root (this script lives in <repo>/sft/).
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── 路径配置 ──────────────────────────────────────────────────────────────────
# NOTE: 基座模型不随仓库分发。原始流程使用本地 gemma-2-2b-it 权重；
# 通过 BASE_MODEL_ID 环境变量指向本地副本，默认从 HuggingFace 拉取。
BASE_MODEL_ID  = os.environ.get("BASE_MODEL_ID", "google/gemma-2-2b-it")
ADAPTER_ID     = str(REPO_ROOT / "sft_output")       # SFT LoRA weights
# NOTE: 合并产物 sft_merged（约 5GB）不随仓库分发，需本地运行本脚本生成。
OUTPUT_DIR     = str(REPO_ROOT / "sft_merged")        # 合并后的完整模型

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1：用 bfloat16 在 CPU 上加载 base model（不量化，才能干净 merge）──
print("=" * 60)
print("Step 1: Loading base model in bfloat16 on CPU...")
print("  预计占用系统内存约 6GB，请确认内存充足。")
print("=" * 60)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
print("  Base model loaded.")

# ── Step 2：加载 SFT LoRA adapter（不可训练，只用于 merge）────────────────
print("\nStep 2: Loading SFT LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_ID, is_trainable=False)
print("  SFT adapter loaded.")

# ── Step 3：merge_and_unload：把 LoRA 的 ΔW 加进基座权重，移除 PEFT 结构 ──
print("\nStep 3: Merging LoRA into base weights (merge_and_unload)...")
merged_model = model.merge_and_unload()
print("  Merge complete. PEFT structure removed.")

# ── Step 4：保存合并后的完整模型 ─────────────────────────────────────────────
print(f"\nStep 4: Saving merged model to {OUTPUT_DIR} ...")
merged_model.save_pretrained(OUTPUT_DIR)
print("  Model weights saved.")

# ── Step 5：保存 tokenizer（直接从 SFT adapter 目录复制，配置一致）──────────
print("\nStep 5: Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print("  Tokenizer saved.")

print("\n" + "=" * 60)
print("合并完成！")
print(f"  输出路径: {OUTPUT_DIR}")
print()
print("  接下来需要修改 train_grpo_interactive.py：")
print("    model_id   = os.environ.get('SFT_BASE_MODEL', str(REPO_ROOT / 'sft_merged'))  # 新基座")
print("    dataset_path = str(REPO_ROOT / 'data' / 'numina_gsm_mix_numeric.jsonl')")
print("    # 删除 adapter_id 相关逻辑，改为从头添加 LoraConfig")
print("=" * 60)
