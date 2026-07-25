---
base_model: sft_merged (local, not distributed)
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- transformers
- grpo
- reinforcement-learning
- math-reasoning
- code-sandbox
license: mit
---

# Model Card — GRPO Checkpoint (GRPO-1800)

## Model Description

LoRA adapter trained with **GRPO (Group Relative Policy Optimization)** under an
execution-guided reward: rollouts write Python in `<STEP>` blocks, a local sandbox
executes them, and a multi-objective reward (correctness + step execution +
self-correction + format/length/repetition penalties) grades each trajectory.
Critic-free, group-relative advantage over **G=4** rollouts per prompt; on-policy
with gradient accumulation (single-GPU VRAM constraint).

- **Developed by:** Ruikang Zhong
- **Model type:** LoRA adapter (PEFT) for causal LM
- **Base model:** `sft_merged` — Gemma-2-2B-it + SFT (merged). **Not distributed**; point `SFT_BASE_MODEL` to your local copy
- **LoRA config:** r=8, alpha=16, dropout=0.05, targets `q/k/v/o/gate/up/down_proj`
- **Language(s):** English (math word problems)
- **License:** MIT

## Training Details

- **Data:** `data/numina_gsm_mix_numeric.jsonl` + `data/hendrycks_math_grpo_numeric.jsonl` (see repository README *Known Issues*: 0% exact duplicates, 2.6% near-duplicates).
- **Algorithm:** GRPO without Critic; advantage = reward − group mean.
- **Reward:** see repository README § Reward Calculation Rules (max correctness 0.9 with exponential step/error discount; C-OPRM LLM salvage ladder 0.0–0.50).
- **Reported as:** "GRPO-1800" in the technical reports.

## Evaluation Results

| Benchmark | This checkpoint | SFT baseline | ZPPO-600 |
|---|---|---|---|
| NuminaMath 500 (accuracy) | 29.20% (146/500) | 29.40% | **32.60%** |
| Hard Math 214, geometry-free (accuracy) | 12.15% (26/214) | 13.08% | **17.29%** |

Notable behavior: on the hard subset the code-generation rate collapses to ~10%
(tool-avoidance under difficulty), which caps its accuracy; ZPPO was trained to
fix exactly this failure mode.

## Intended Use

Research on execution-guided RL for small language models; reproducibility
baseline for GRPO vs ZPPO comparisons. **Not** intended for production math
solving or general chat.

## Limitations

- Tool-avoidance on hard problems (code rate ~10%), no accuracy gain over SFT.
- Evaluation set contains 2 near-exact duplicates (≥0.99 similarity) of training-segment questions, plus 16 template variants at ≥0.90 (disclosed in repo README *Known Issues*).
- Requires the `<STEP>`/sandbox protocol; outputs raw code blocks under distribution shift.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("<path-to>/sft_merged", device_map="auto")
model = PeftModel.from_pretrained(base, "weights/grpo_checkpoint")
tok = AutoTokenizer.from_pretrained("weights/grpo_checkpoint")
```

### Framework versions

- PEFT 0.19.1
