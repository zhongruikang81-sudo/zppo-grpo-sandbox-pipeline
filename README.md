# ZPPO & GRPO Mathematical Reinforcement Learning Pipeline

This repository consolidates the entire code, scripts, configurations, and datasets for training and benchmarking **GRPO** and **ZPPO** (Zone of Proximal Policy Optimization) algorithms. It evaluates language models on high-school level mathematical word problems utilizing a local **Python Sandbox** for execution-guided learning.

---

## 📁 Directory Structure

```
zppo_grpo_sandbox_pipeline/
├── README.md                 # Project documentation & reward rules
├── requirements.txt          # Python dependencies
├── core/
│   ├── sandbox.py            # Local code sandbox (executes python, catches traceback, timeout & input() stub)
│   └── evaluator.py          # Answer equivalence, multi-objective reward, LLM-assisted process scoring
├── sft/                      # Stage 1 — SFT base construction
│   ├── generate_react_dataset.py  # Teacher-generated ReAct-style trajectories (DeepSeek API)
│   ├── train_sft.py          # SFT LoRA training (BASE_MODEL_ID)
│   └── merge_sft.py          # Merge LoRA into base → sft_merged/
├── dpo/                      # Stage 2 — DPO alignment (format-collapse & anti-repetition repair)
│   ├── generate_dpo_pairs.py      # Pair construction: clean-vs-repetitive correct trajectories
│   ├── build_dpo_dataset_v3.py    # LLM-judged pair building
│   ├── clean_dpo_dataset.py       # Pair filtering / cleaning
│   ├── train_dpo_formatting.py    # DPO on formatting pairs
│   └── train_dpo_anti_repetition.py
├── grpo/                     # Stage 3 — GRPO
│   ├── train_grpo.py         # Interactive GRPO loop (critic-free, G=4, gradient accumulation)
│   ├── train_grpo_interactive_v2.py
│   └── auto_recovery.py      # Checkpoint auto-recovery
├── zppo/                     # Stage 4 — ZPPO
│   ├── train_zppo.py         # ZPPO LoRA training + Prompt Replay Buffer + Zero-Advantage Skip
│   ├── zppo_buffer.py        # BCQ prompting & graduation replay buffer
│   └── zppo_interactive_teacher.py   # In-context teacher rollout generator
├── evaluation/
│   ├── run_bench500.py       # 500-question comparative benchmark (salvage counting threshold ≥ 0.40)
│   └── run_hard_math.py      # 214-question geometry-free Hard Math subset benchmark
├── tests/                    # Unit tests: sandbox, evaluator, GRPO agent
├── docs/
│   ├── zppo_grpo_sandbox_comprehensive_walkthrough.md # Detailed training pipeline walkthrough
│   └── zppo_alignment_comprehensive_final_report.md   # Final 500-question & 214-question results report
├── results/
│   ├── bench500_*.json       # Raw evaluation trajectories for the 500-question run
│   └── bench250_hard_*.json  # Raw evaluation trajectories for the hard-subset run
├── weights/
│   ├── grpo_checkpoint/      # Trained GRPO adapter weights (without optimizer.pt)
│   ├── zppo_checkpoint_600/  # Trained ZPPO-600 adapter weights (sweet spot)
│   └── zppo_checkpoint_960/  # Trained ZPPO-960 adapter weights
├── scripts/
│   ├── run_pipeline.ps1      # PowerShell execution wrapper
│   └── run_zppo_recovery.ps1 # ZPPO checkpoint resume manager
└── data/
    ├── numina_gsm_mix_numeric.jsonl      # NuminaMath numeric subset (training)
    ├── hendrycks_math_grpo_numeric.jsonl # MATH numeric subset (training)
    ├── numina_500_test.json              # 500-question held-out test set (see Known Issues)
    ├── hard_math_250_test.json           # Hard Math subset (250 questions)
    ├── hard_math_240_no_geom_test.json   # Geometry-free hard subset (240 questions)
    ├── extract_500_test.py               # Script that extracted the 500-question test set
    ├── dpo_formatting_alignment.jsonl    # DPO pairs: format-collapse repair
    ├── dpo_anti_repetition.jsonl         # DPO pairs: anti-repetition (deduplicated)
    └── dpo_anti_repetition_original.jsonl# Raw anti-repetition pairs (pre-dedup)
```

> **Notes on data, models, and API credentials**
> - The merged SFT base model `sft_merged` (~5GB) is **not** distributed with this repository; point the `SFT_BASE_MODEL` environment variable at your local merged model directory (the DPO/SFT scripts likewise take `--model-id` / `BASE_MODEL_ID`).
> - LLM-assisted scoring and teacher rollouts require the `DEEPSEEK_API_KEY` environment variable. The judge/teacher model is read from `DEEPSEEK_JUDGE_MODEL` (default: `deepseek-chat`). The original hardcoded judge model name `deepseek-v4-pro` is **not a valid DeepSeek model name** — PRM salvaging requires a valid model name.

---

## ⚙️ Reward Calculation Rules (Reward Design)

Both GRPO and ZPPO training pipelines share the same **execution-guided, multi-objective reward** implemented in `core/evaluator.py::evaluate_multi_turn_completion`. The total reward is:

```
total = avg_step_reward + correctness_reward + self_correction_bonus
        + format_penalty + first_step_penalty + unclosed_penalty
        + length_penalty − repetition_penalty
```

1. **Step execution shaping (`avg_step_reward`)**: each `<STEP>` code block is AST-checked and executed **cumulatively** in the sandbox. `+0.1` per block that parses and runs, `−0.1` per failing block; averaged over blocks.
2. **Self-correction bonus**: `+0.2` if a block succeeds after a previous block failed (debugging recovery).
3. **Format penalties**:
   - No `<STEP>` block at all → immediate return `−0.3` (+ length/format/first-step penalties).
   - Raw ` ```python ` block outside `<STEP>` (protocol bypass) → `−0.15`.
   - First turn contains no `<STEP>` → `−0.5` (mandatory tool-use on turn 1).
   - Unclosed `<STEP>` tag → `−0.1`.
4. **Length penalty**: `−0.00001 × len(completion_text)` (anti-rambling).
5. **Repetition penalty**: `−0.2` if the text contains ≥25 identical consecutive chars, a 7-streak repeated character pattern, or ≥5 identical consecutive lines (anti format-collapse).
6. **Correctness reward (max 0.9, exponentially discounted)**:
   - Final sandbox stdout is compared against the target answer with a sympy-backed comparator (`compare_math_answers`), gated by a hard-coding/cheat detector (`is_suspicious_code`).
   - If correct: `correctness_reward = 0.9 × discount`, where `discount = 0.8^errors × 0.9^(blocks−1)` — earlier correctness and error-free trajectories are worth more.
7. **LLM-assisted process salvaging (C-OPRM, alleviates reward sparsity)**:
   - If symbolic match fails (and format is valid), the pipeline calls the **LLM process judge** (DeepSeek API), which grades the trajectory on a discrete ladder: `0.0 / 0.15 / 0.30 / 0.40 / 0.50`.
   - `correctness_reward = salvage_score × (0.9 / 0.5) × discount` — a judge score of `0.50` rescales to the full `0.9`; lower scores map linearly.
   - If `DEEPSEEK_API_KEY` is unset, salvaging degrades gracefully to score `0.0`.

**Group-relative advantage (GRPO-style, critic-free)**

- Each prompt is sampled **G=4** rollouts (`train_grpo.py`, `auto_recovery.py`). Advantage is computed group-relatively without a Critic network:
  $$A_i = R_i - \bar{R}$$
- Due to VRAM limits the pipeline is **on-policy with gradient accumulation**: rollouts are graded, then per-sample `backward` calls accumulate gradients before a single optimizer step.
- **Zero-Advantage Skip (ZPPO only)**: in `zppo/train_zppo.py`, if all G rollouts receive identical rewards (advantage = 0), the backward pass is skipped entirely to save GPU cycles. The plain GRPO trainer does **not** implement this skip.

> **Salvage threshold note (canonical statement).** During *training*, salvage is a continuous rescale capped at `0.50 → full marks`; there is no binary threshold. During *evaluation* (`run_bench500.py`, `run_hard_math.py`), a trajectory is additionally *counted as correct* when its judge salvage score is **≥ 0.40** (the "minor slip" tier), which is what the benchmark reports state (`SALVAGE_THRESH = 0.40`). The two numbers refer to different stages and are intentional.

---

## 🚀 How to Run

### 1. Requirements
Ensure the following dependencies are installed in your PyTorch environment:
```bash
pip install torch transformers peft accelerate bitsandbytes sympy pandas pyarrow requests
```

## 🚀 Reproduction Guide

### 1. Requirements

```bash
pip install torch transformers peft accelerate bitsandbytes sympy pandas pyarrow requests openai scipy numpy
```

The sandbox needs write permission in a local scratch directory: `core/sandbox.py` writes temporary python scripts, executes them with a timeout, and captures stdout/stderr.

### 2. Environment variables

| Variable | Required by | Default | Purpose |
|---|---|---|---|
| `SFT_BASE_MODEL` | `grpo/train_grpo*.py`, `grpo/auto_recovery.py`, `zppo/train_zppo.py`, `evaluation/run_*.py` | `./sft_merged` | Path to the merged SFT base model (~5GB, **not** distributed with this repo) |
| `BASE_MODEL_ID` | `sft/train_sft.py`, `sft/merge_sft.py`, `tests/` | `google/gemma-2-2b-it` | Base model for SFT stage |
| `DEEPSEEK_API_KEY` | `core/evaluator.py`, `zppo/zppo_interactive_teacher.py`, `dpo/build_dpo_dataset_v3.py`, `sft/generate_react_dataset.py` | — (unset) | DeepSeek API key for LLM-assisted process scoring & teacher rollouts. If unset, PRM salvaging degrades gracefully to `0.0` and teacher-dependent stages are skipped |
| `DEEPSEEK_JUDGE_MODEL` | same as above | `deepseek-chat` | Judge/teacher model name. Must be a **valid** DeepSeek model name — the original hardcoded `deepseek-v4-pro` was not valid and silently disabled salvaging |

Example (PowerShell):

```powershell
$env:SFT_BASE_MODEL   = "E:\models\sft_merged"
$env:DEEPSEEK_API_KEY = "sk-..."        # your own key
$env:DEEPSEEK_JUDGE_MODEL = "deepseek-chat"
```

### 3. Data notes

- `data/numina_gsm_mix_numeric.jsonl`, `data/hendrycks_math_grpo_numeric.jsonl` — training question pools (see *Known Issues* for duplication rate).
- `data/numina_500_test.json` — held-out 500-question evaluation set, produced by `data/extract_500_test.py` (see *Known Issues* for 2 overlapping questions).
- `data/hard_math_250_test.json` / `hard_math_240_no_geom_test.json` — hard subsets; the 240 file excludes geometry (`[asy]`) items, benchmarked as the 214-question geometry-free set after filtering.
- `data/dpo_*.jsonl` — DPO preference pairs. `dpo_anti_repetition.jsonl` is the deduplicated version of `dpo_anti_repetition_original.jsonl`; `dpo_formatting_alignment.jsonl` contains format-collapse repair pairs (chosen = clean correct, rejected = repetitive).

### 4. Full pipeline (four stages)

```bash
# Stage 1 — SFT base (produces sft_merged/)
python sft/generate_react_dataset.py      # needs DEEPSEEK_API_KEY
python sft/train_sft.py                   # uses BASE_MODEL_ID
python sft/merge_sft.py

# Stage 2 — DPO alignment (format-collapse & anti-repetition repair)
python dpo/generate_dpo_pairs.py
python dpo/build_dpo_dataset_v3.py        # needs DEEPSEEK_API_KEY
python dpo/clean_dpo_dataset.py
python dpo/train_dpo_formatting.py
python dpo/train_dpo_anti_repetition.py

# Stage 3 — GRPO (critic-free, G=4, on-policy + gradient accumulation)
python grpo/train_grpo.py                 # or scripts/run_pipeline.ps1

# Stage 4 — ZPPO (replay buffer + in-context teacher + zero-advantage skip)
./scripts/run_zppo_recovery.ps1           # handles checkpoint resume automatically
```

### 5. Running benchmarks

```bash
python evaluation/run_bench500.py         # 500-question comparative benchmark
python evaluation/run_hard_math.py        # 214-question geometry-free hard subset
```

Both scripts resume from partially written `results/bench*.json` caches after interruptions, and emit a comparison report (`hard_math_benchmark_report.md` for the hard subset) with geometry items filtered out.

### 6. Tests

```bash
python -m unittest discover tests         # 14 tests: sandbox, evaluator, GRPO agent
```

Note: `tests/test_grpo_agent.py` imports `torch` and must run inside your PyTorch environment; the sandbox/evaluator suites (13 tests) run anywhere.

### 7. Checkpoints

Adapter weights for GRPO, ZPPO-600 (sweet spot) and ZPPO-960 live under `weights/`; each directory carries its own model card (`README.md`) describing base model, training config and known limitations.

---

## ⚠️ Known Issues (Data & Evaluation)

All figures below are reproducible with `data/audit_data_overlap.py`
(similarity = `difflib.SequenceMatcher` on whitespace-normalized prompts).

1. **Test set is a slice of the training source file (fragile holdout, 0 exact leakage).**
   `data/numina_500_test.json` is indices 1980–2479 of `numina_gsm_mix_numeric.jsonl`.
   Holdout validity relies on training consumption staying below index 1980:
   GRPO samples deterministically (`mixed_dataset[step % len]`, max step 1800 →
   indices ≤ 1799) and the ZPPO replay buffer consumes sequentially from index 0
   (960 steps, far below 1980). Measured exact-match leakage against the
   consumed segment: **0 questions**. However, **no code-level guard enforces the
   boundary** — extending training past index 1980 would silently leak. A future
   fix is to hard-filter indices ≥ 1980 at dataset load time in the trainers.

2. **Near-duplicate semantic leakage: 2 near-exact + 16 template variants.**
   2 test questions have ≥ 0.99 similarity twins inside the training segment
   (test#83↔train#962, test#295↔train#165; differing only in minor formatting),
   and 16 more have ≥ 0.90 similarity template variants (same problem, different
   numbers). Worst-case inflation of the 500-question accuracy is
   18/500 ≈ 3.6pp; the real effect is smaller because variants differ
   numerically. A future rebuild of the test set should exclude items with
   ≥ 0.85 similarity to any training-segment question.

3. **Training-pool near-duplication: 2.6% (not 22%).**
   Exact duplicates: 0/2500. Near-duplicates (≥ 0.90 similarity with a shared
   60-char prefix): 64/2500 rows (2.6%, 34 pairs) — mostly GSM-style template
   variants, which mildly over-sample those templates during training. (A naive
   prefix-signature count yields 22.6%, but that figure is dominated by shared
   instruction boilerplate and overstates the true rate.)

4. **ZPPO-960 late-stage regression.** Debug-salvage rate collapses from
   15.22% (step 600) to 4.35% (step 960) on the hard subset — see
   `weights/zppo_checkpoint_960/README.md` and
   `docs/zppo_alignment_comprehensive_final_report.md`.

5. **GRPO importance-sampling ratio — FIXED (2026-07-25).**
   Previously, the PPO ratio denominator in the GRPO trainers used
   log-probabilities from the fixed SFT base instead of the rollout policy
   π_θ_old. This has been corrected in all three GRPO entry points
   (`grpo/train_grpo.py`, `grpo/train_grpo_interactive_v2.py`,
   `grpo/auto_recovery.py`): the ratio is now `exp(logp_active − logp_old)`
   with π_θ_old captured (adapter enabled) before each weight update, while the
   SFT base is used only for the KL-to-reference regularizer, matching textbook
   PPO/GRPO. `zppo/train_zppo.py` was already correct (it stores
   `token_lp_old` at rollout time). **Caveat:** the published checkpoints under
   `weights/` and the numbers in `docs/` were produced under the old objective;
   the fix applies to future training runs.
