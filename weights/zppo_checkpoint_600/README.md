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

# Model Card — ZPPO Checkpoint 600 (Sweet Spot)

## Model Description

LoRA adapter trained with **ZPPO (Zone of Proximal Policy Optimization)**, step 600 —
the best-performing checkpoint in this project. ZPPO extends GRPO with a
BCQ-style reflection curriculum, a prompt replay buffer with graduation, an
in-context teacher (higher reward frequency on hard problems), and
**Zero-Advantage Skip** (skips backward when all G rollouts tie).

- **Developed by:** Ruikang Zhong
- **Model type:** LoRA adapter (PEFT) for causal LM
- **Base model:** `sft_merged` — Gemma-2-2B-it + SFT (merged). **Not distributed**; point `SFT_BASE_MODEL` to your local copy
- **LoRA config:** r=8, alpha=16, dropout=0.05, targets `q/k/v/o/gate/up/down_proj`
- **Language(s):** English (math word problems)
- **License:** MIT

## Training Details

- **Data:** `data/numina_gsm_mix_numeric.jsonl` + `data/hendrycks_math_grpo_numeric.jsonl` (see repository README *Known Issues*: 0% exact duplicates, 2.6% near-duplicates).
- **Algorithm:** ZPPO over the GRPO-trained adapter lineage; G=4 group-relative advantage, execution-guided multi-objective reward (see repo README § Reward Calculation Rules), replay buffer state in `zppo_prb_state.json`.
- **Reported as:** "ZPPO-600 (中期)" in the technical reports.

## Evaluation Results

| Benchmark | This checkpoint | SFT | GRPO-1800 | ZPPO-960 |
|---|---|---|---|---|
| NuminaMath 500 (accuracy) | **32.60%** (163/500) 🏆 | 29.40% | 29.20% | 32.60% |
| Hard Math 214, geometry-free (accuracy) | **17.29%** (37/214) 🏆 | 13.08% | 12.15% | 14.95% |

Behavioral effects (see `docs/zppo_alignment_comprehensive_final_report.md`):

- Code-generation rate pushed from 28% (SFT) to **70%+** on the mixed set, and
  anchored at **~50%** on the hard subset where SFT/GRPO collapse to 10–14%.
- Reflection training transfers to pure-text reasoning: non-coding accuracy
  17.43% vs SFT 13.04% / GRPO 11.46%.
- Debug-salvage rate after first-turn errors: 15.22% (collapses to 4.35% at step 960).

## Intended Use

Research checkpoint demonstrating execution-guided preference curricula for
small models. **Not** intended for production math solving or general chat.

## Limitations

- 500-question evaluation set contains 2 near-exact duplicates (≥0.99 similarity) of training-segment questions, plus 16 template variants at ≥0.90 (disclosed in repo README *Known Issues*).
- Trained on English numeric-answer math problems only; geometry (`[asy]`) items are explicitly out of scope and filtered at evaluation.
- Requires the `<STEP>`/sandbox protocol for full behavior.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("<path-to>/sft_merged", device_map="auto")
model = PeftModel.from_pretrained(base, "weights/zppo_checkpoint_600")
tok = AutoTokenizer.from_pretrained("weights/zppo_checkpoint_600")
```

### Framework versions

- PEFT 0.19.1
