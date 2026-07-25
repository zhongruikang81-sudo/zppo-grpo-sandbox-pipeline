---
base_model: sft_merged (local, not distributed)
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- transformers
- zppo
- reinforcement-learning
- math-reasoning
- code-sandbox
license: mit
---

# Model Card — ZPPO Checkpoint 960 (Late-Stage, With Known Regression)

## Model Description

LoRA adapter trained with **ZPPO (Zone of Proximal Policy Optimization)**, step 960 —
the final checkpoint. It matches ZPPO-600 on the mixed benchmark but shows a
**documented late-stage regression** on hard problems: debugging recovery
collapses, indicating over-alignment rigidity from extended RL. Released for
analysis of RL over-training effects; for best performance use
`zppo_checkpoint_600`.

- **Developed by:** Ruikang Zhong
- **Model type:** LoRA adapter (PEFT) for causal LM
- **Base model:** `sft_merged` — Gemma-2-2B-it + SFT (merged). **Not distributed**; point `SFT_BASE_MODEL` to your local copy
- **LoRA config:** r=8, alpha=16, dropout=0.05, targets `q/k/v/o/gate/up/down_proj`
- **Language(s):** English (math word problems)
- **License:** MIT

## Training Details

- Same pipeline as ZPPO-600, trained 360 steps further (see repo README § Reward Calculation Rules and `docs/` reports).
- **Reported as:** "ZPPO-960 (后期)" in the technical reports.

## Evaluation Results

| Benchmark | This checkpoint | ZPPO-600 |
|---|---|---|
| NuminaMath 500 (accuracy) | **32.60%** (163/500) | **32.60%** |
| Hard Math 214, geometry-free (accuracy) | 14.95% (32/214) | **17.29%** |
| First-turn accuracy (hard subset) | **5.61%** (12/214) | 5.14% |
| Debug-salvage rate (hard subset) | 4.35% (2/46) ⚠️ | **15.22%** (7/46) |

Known regression: after a first-turn code error, this checkpoint recovers via
multi-turn debugging only 4.35% of the time (vs 15.22% at step 600) — debugging
logic becomes stereotyped and falls into repetitive correction loops, a
signature of over-alignment bias from extended RL.

## Intended Use

Research on RL over-training / alignment drift. **Not** recommended as the
deployment checkpoint (use step 600), and not intended for production math
solving or general chat.

## Limitations

- Documented debug-recovery collapse on hard problems (see above).
- 500-question evaluation set contains 2 near-exact duplicates (≥0.99 similarity) of training-segment questions, plus 16 template variants at ≥0.90 (disclosed in repo README *Known Issues*).
- Training pool: 0% exact duplicates, 2.6% near-duplicates (measured; see repo README *Known Issues*).
- Requires the `<STEP>`/sandbox protocol; English numeric-answer math only.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("<path-to>/sft_merged", device_map="auto")
model = PeftModel.from_pretrained(base, "weights/zppo_checkpoint_960")
tok = AutoTokenizer.from_pretrained("weights/zppo_checkpoint_960")
```

### Framework versions

- PEFT 0.19.1
