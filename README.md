# ZPPO & GRPO Mathematical Reinforcement Learning Pipeline

This repository consolidates the entire code, scripts, configurations, and datasets for training and benchmarking **GRPO** and **ZPPO** (Zone of Proximal Policy Optimization) algorithms. It evaluates language models on high-school level mathematical word problems utilizing a local **Python Sandbox** for execution-guided learning.

---

## 📁 Directory Structure

```
zppo_grpo_sandbox_pipeline/
├── README.md                 # Project documentation & reward rules
├── docs/
│   ├── zppo_grpo_sandbox_comprehensive_walkthrough.md # Detailed training pipeline walkthrough
│   └── zppo_alignment_comprehensive_final_report.md  # Final 500-question & 214-question experimental results report
├── results/
│   ├── bench500_*.json       # Raw evaluation trajectories and outputs for 500-question run
│   └── bench250_hard_*.json  # Raw evaluation trajectories and outputs for 250-question run
├── weights/
│   ├── grpo_checkpoint/      # Trained GRPO adapter weights (without optimizer.pt)
│   ├── zppo_checkpoint_600/  # Trained ZPPO-600 adapter weights (Sweet Spot)
│   └── zppo_checkpoint_960/  # Trained ZPPO-960 adapter weights
├── core/
│   ├── sandbox.py            # Local code sandbox (executes python and catches traceback)
│   └── evaluator.py          # Math answer equivalency & 大模型辅助过程打分 judge interface
├── grpo/
│   ├── train_grpo.py         # Interactive GRPO training loop (no Critic network)
│   └── auto_recovery.py      # Automated GRPO training checkpoint recovery
├── zppo/
│   ├── train_zppo.py         # ZPPO LoRA training with Prompt Replay Buffer
│   ├── zppo_buffer.py        # Replay buffer handling BCQ prompting & graduation
│   └── zppo_interactive_teacher.py   # Teacher rollout generator
├── evaluation/
│   ├── run_bench500.py       # 500-question comparative benchmark evaluator
│   └── run_hard_math.py      # 215-question Geometry-Free Hard Math subset benchmark
├── scripts/
│   ├── run_pipeline.ps1      # PowerShell execution wrapper
│   └── run_zppo_recovery.ps1 # ZPPO checkpoint resume manager
└── data/
    ├── numina_gsm_mix_numeric.jsonl     # NuminaMath numeric subset for training
    ├── hendrycks_math_grpo_numeric.jsonl # MATH dataset numeric subset for training
    └── hard_math_240_no_geom_test.json   # 240-question Hard Math subset
```

---

## ⚙️ Reward Calculation Rules (Reward Design)

Both GRPO and ZPPO training pipelines use an **Execution-guided + Process-level & Step-level rewards Reward system** to grade rollouts:

1. **Local Symbolic Match (Equivalence Check)**:
   - The final generated response is parsed for mathematical answer extraction.
   - It is compared against the target answer using a sympy-backed comparator (`compare_math_answers`).
   - If it matches symbolically: **Reward = 1.0, Correct = True**.

2. **LLM-assisted Process Scoring & Salvaging (to alleviate reward sparsity)**:
   - If the local equivalence check fails (due to minor formatting differences or rounding), the pipeline calls the **LLM-assisted Process scoring API**.
   - If the PRM judge score is **$\ge 0.50$**: **Reward = 1.0, Correct = True (Salvaged)**.
   - If the PRM judge score is **$< 0.50$**: **Reward = Process_Score * 2.0 (Scaled 0.0 to 1.0), Correct = False**.

3. **Format & Tool-use Penalty**:
   - If the response contains no Python code blocks (no `<STEP> ... </STEP>`), it is penalized: **Reward = 0.0, Correct = False**.

4. **Group-Relative Advantage (GRPO-style)**:
   - A single prompt is sampled **$G=2$ times** (Rollout 1 and Rollout 2).
   - Advantage is calculated group-relatively without a Critic network:
     $$A_i = R_i - ar{R}$$
   - **Zero-Advantage Skip**: If $R_1 == R_2$ (Advantage is 0 for both rollouts), the training step is skipped entirely to prevent gradient collapse and save GPU cycles.

---

## 🚀 How to Run

### 1. Requirements
Ensure the following dependencies are installed in your PyTorch environment:
```bash
pip install torch transformers peft accelerate bitsandbytes sympy pandas pyarrow requests
```

### 2. Sandbox Execution Setup
Make sure the environment has write permissions in the local directory, as `sandbox.py` creates temporary scripts to execute python code and captures standard stdout/stderr outputs.

### 3. ZPPO LoRA Training
To start or resume the ZPPO training pipeline:
```powershell
./scripts/run_zppo_recovery.ps1
```
This script handles LoRA checkpoint loading and loads/saves the prompt replay buffer state (`zppo_prb_state.json`) periodically to allow resuming training transparently.

### 4. Running Benchmarks
To evaluate your models (SFT, GRPO, ZPPO checkpoints) on the 215-question Geometry-free MATH subset:
```bash
python evaluation/run_hard_math.py
```
This outputs individual JSON log caches and automatically generates a clean comparison report `hard_math_benchmark_report.md` filtering out any Geometry visualization questions.
