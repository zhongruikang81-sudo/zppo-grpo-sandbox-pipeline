"""
Hard Math Subset Benchmark (SFT vs GRPO-1800 vs ZPPO-600 vs ZPPO-960)
- Evaluates 4 models sequentially on the 250-question Hard Math Subset
- Output files: bench250_hard_{model}.json
- Automatically generates hard_math_benchmark_report.md
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import re
import gc
import json
import time
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    StoppingCriteria, StoppingCriteriaList,
    LogitsProcessor, LogitsProcessorList,
)
from peft import PeftModel

os.environ["HF_ENDPOINT"]           = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(r"E:\math workspace")
from sandbox       import execute_accumulated_code
from evaluator_v2  import compare_math_answers, get_prm_salvage_score

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_MODEL   = r"E:\math workspace\sft_merged"
GRPO_ADAPTER = r"E:\math workspace\grpo_output_discounted\checkpoint_step1800_v3"
ZPPO_600_DIR = r"E:\math workspace\grpo_output_discounted\zppo_checkpoint\checkpoint_step600"
ZPPO_960_DIR = r"E:\math workspace\grpo_output_discounted\zppo_checkpoint\checkpoint_step960"

SCRATCH = r"C:\Users\rick john\.gemini\antigravity\brain\8ceedcb6-148d-478c-b186-c0bb494fe889\scratch"
TEST_SET = os.path.join(SCRATCH, "hard_math_250_test.json")

MAX_TURNS      = 6
MAX_NEW_TOKENS = 1024
SALVAGE_THRESH = 0.40

class SuppressConsecutiveNewlines(LogitsProcessor):
    def __init__(self, tokenizer):
        self.newline_tokens = {108, 109, 110}
        self.tokenizer = tokenizer
    def __call__(self, input_ids, scores):
        for i in range(input_ids.shape[0]):
            last = self.tokenizer.decode(input_ids[i][-8:].tolist())
            if re.search(r'\n\s*$', last):
                for t in self.newline_tokens:
                    scores[i, t] = float("-inf")
            scores[i, 109] = float("-inf")
            scores[i, 110] = float("-inf")
        return scores

class SuppressRepetitiveContent(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        self.eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    def __call__(self, input_ids, scores):
        for i in range(input_ids.shape[0]):
            seq = input_ids[i]
            lb  = min(150, len(seq))
            if lb < 10: continue
            txt = self.tokenizer.decode(seq[-lb:].tolist())
            lines = [l.strip() for l in txt.split("\n") if l.strip()]
            if len(lines) >= 3 and lines[-1] == lines[-2] == lines[-3]:
                scores[i, :] = float("-inf")
                scores[i, self.eos] = 0.0
                if isinstance(self.eot, int) and self.eot != self.tokenizer.unk_token_id:
                    scores[i, self.eot] = 0.0
                continue
            if len(txt) >= 15:
                suf = txt[-15:]
                if len(set(suf)) == 1 and suf[0] in ".-_*+=#~":
                    scores[i, :] = float("-inf")
                    scores[i, self.eos] = 0.0
                    if isinstance(self.eot, int) and self.eot != self.tokenizer.unk_token_id:
                        scores[i, self.eot] = 0.0
        return scores

class StopOnStepEnd(StoppingCriteria):
    def __init__(self, stop_ids, tokenizer=None):
        self.stop_ids = list(stop_ids)
        self.n = len(self.stop_ids)
        self.tokenizer = tokenizer
    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0]
        if len(seq) >= self.n and list(seq[-self.n:].cpu().numpy()) == self.stop_ids:
            return True
        if self.tokenizer and len(seq) >= 8:
            if self.tokenizer.decode(seq[-8:]).endswith("\n\n\n\n"):
                return True
        return False

def evaluate(model, tokenizer, questions, out_path, model_name):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")

    stop_ids = tokenizer.encode("</STEP>", add_special_tokens=False)
    eot_id   = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids  = [tokenizer.eos_token_id]
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id:
        eos_ids.append(eot_id)

    results = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"  Loaded {len(results)} existing results, resuming...")
        except:
            results = []

    model.eval()
    model.config.use_cache = True

    for idx, q_data in enumerate(questions):
        if idx < len(results):
            continue

        prompt_text = q_data["prompt"]
        target      = str(q_data["answer"])
        print(f"\n  [{model_name}] Q{idx+1}/250 | Type: {q_data['type']} | Target: {target[:40]}")

        sys_prompt = (
            "You are a mathematical assistant. Use Python code wrapped inside <STEP>\n"
            "```python\n# code\n```\n</STEP> to solve the problem. "
            "Note: Never use input() in Python code. "
            "Ensure all variables are defined before use in the current step.\n\n"
            + prompt_text
        )
        history = [{"role": "user", "content": sys_prompt}]

        code_gen_count = 0
        exec_success_count = 0
        turn_details = []

        with torch.no_grad():
            for turn in range(MAX_TURNS):
                prompt_str = tokenizer.apply_chat_template(
                    history, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt_str, return_tensors="pt").to("cuda")

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=1.2,
                    eos_token_id=eos_ids,
                    stopping_criteria=StoppingCriteriaList([StopOnStepEnd(stop_ids, tokenizer)]),
                    logits_processor=LogitsProcessorList([
                        SuppressConsecutiveNewlines(tokenizer),
                        SuppressRepetitiveContent(tokenizer),
                    ]),
                )

                gen_ids       = outputs[0][inputs.input_ids.shape[1]:]
                assistant_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
                history.append({"role": "assistant", "content": assistant_txt})

                has_code       = "</STEP>" in assistant_txt
                exec_ok        = False
                observation    = ""

                if has_code:
                    code_gen_count += 1
                    ok, stdout, stderr = execute_accumulated_code(history)
                    observation = (stdout.strip() if ok else stderr.strip()) or (
                        "[Success: No output]" if ok else "[Error: Unknown]"
                    )
                    if len(observation) > 1000:
                        observation = observation[:1000] + "\n... [Observation Truncated] ..."
                    exec_ok = ok and not any(
                        e in observation for e in ["Error", "Traceback", "SyntaxError", "NameError", "TypeError"]
                    )
                    if exec_ok:
                        exec_success_count += 1
                    history.append({"role": "user", "content": f"Observation:\n{observation}\n"})

                turn_details.append({
                    "turn": turn + 1,
                    "text": assistant_txt,
                    "has_code": has_code,
                    "observation": observation,
                    "executed_success": exec_ok,
                })

                if not has_code:
                    break

        comp_parts = []
        for msg in history[1:]:
            if msg["role"] == "assistant":
                comp_parts.append(msg["content"])
            else:
                comp_parts.append(f"\nObservation:\n{msg['content']}\n")
        final_completion = "".join(comp_parts)

        matched = compare_math_answers(final_completion, target)
        salvage_score = 0.0
        is_salvaged = False

        if not matched:
            try:
                salvage_score = get_prm_salvage_score(final_completion, target, prompt_text)
                if salvage_score >= SALVAGE_THRESH:
                    matched = True
                    is_salvaged = True
            except Exception as e:
                print(f"    PRM salvage failed: {e}")
                salvage_score = -1

        results.append({
            "index": idx,
            "prompt": prompt_text,
            "target": target,
            "type": q_data["type"],
            "level": q_data["level"],
            "matched": matched,
            "is_salvaged": is_salvaged,
            "salvage_score": salvage_score,
            "has_code_gen": code_gen_count > 0,
            "code_blocks_generated": code_gen_count,
            "code_blocks_success": exec_success_count,
            "turns": turn_details,
            "final_completion": final_completion
        })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"    Done Q{idx+1} | Code blocks: {code_gen_count} | Correct: {matched}")
        torch.cuda.empty_cache()

    return results

def get_stats(data):
    total   = len(data)
    correct = sum(1 for x in data if x["matched"])
    code_q  = sum(1 for x in data if x["has_code_gen"])
    tot_blk = sum(x["code_blocks_generated"] for x in data)
    ok_blk  = sum(x["code_blocks_success"]   for x in data)
    salvaged = sum(1 for x in data if x.get("is_salvaged", False))
    
    # Stratified breakdown
    by_type = {}
    for x in data:
        t = x["type"]
        if t not in by_type:
            by_type[t] = {"total": 0, "correct": 0}
        by_type[t]["total"] += 1
        if x["matched"]:
            by_type[t]["correct"] += 1
            
    return {
        "total":    total,
        "correct":  correct,
        "code_q":   code_q,
        "tot_blk":  tot_blk,
        "ok_blk":   ok_blk,
        "salvaged": salvaged,
        "by_type":  by_type
    }

def make_final_report(sft, grpo, z600, z960):
    # Filter out Geometry questions in-place to exclude visualization noise
    sft  = [x for x in sft  if x.get("type", "").lower() != "geometry"]
    grpo = [x for x in grpo if x.get("type", "").lower() != "geometry"]
    z600 = [x for x in z600 if x.get("type", "").lower() != "geometry"]
    z960 = [x for x in z960 if x.get("type", "").lower() != "geometry"]

    s  = get_stats(sft)
    g  = get_stats(grpo)
    z6 = get_stats(z600)
    z9 = get_stats(z960)

    def pct(a, b): return f"{a/b*100:.2f}%" if b else "N/A"

    lines = []
    lines.append("# SFT vs GRPO-1800 vs ZPPO-600 vs ZPPO-960 | Hard Math Subset (Geometry-Free, 215题) 终极对比报告\n")
    lines.append(f"> 测试集：MATH (Level 3 & 4) 250题切片剔除 35 道几何题，保留 215 道纯计算题平衡集 | 答案限制：Numeric | 解码：贪婪\n")
    lines.append("## 一、 核心对齐指标\n")

    lines.append("| 指标 | SFT Baseline | GRPO-1800 | ZPPO-600 (中期) | ZPPO-960 (后期) | 性能变化 (960 vs 600) |")
    lines.append("|---|---|---|---|---|---|")
    
    diff_val = (z9['correct']/z9['total'] - z6['correct']/z6['total']) * 100
    lines.append(
        f"| **准确率 Accuracy** | {pct(s['correct'],s['total'])} ({s['correct']}/{s['total']}) "
        f"| {pct(g['correct'],g['total'])} ({g['correct']}/{g['total']}) "
        f"| {pct(z6['correct'],z6['total'])} ({z6['correct']}/{z6['total']}) "
        f"| {pct(z9['correct'],z9['total'])} ({z9['correct']}/{z9['total']}) "
        f"| {diff_val:+.2f}% |"
    )
    
    diff_code = (z9['code_q']/z9['total'] - z6['code_q']/z6['total']) * 100
    lines.append(
        f"| **写码率 Code Gen** | {pct(s['code_q'],s['total'])} "
        f"| {pct(g['code_q'],g['total'])} "
        f"| {pct(z6['code_q'],z6['total'])} "
        f"| {pct(z9['code_q'],z9['total'])} "
        f"| {diff_code:+.2f}% |"
    )
    
    diff_exec = (z9['ok_blk']/z9['tot_blk'] - z6['ok_blk']/z6['tot_blk']) * 100 if z6['tot_blk'] and z9['tot_blk'] else 0.0
    lines.append(
        f"| **代码执行成功率** | {pct(s['ok_blk'],s['tot_blk'])} ({s['ok_blk']}/{s['tot_blk']}) "
        f"| {pct(g['ok_blk'],g['tot_blk'])} ({g['ok_blk']}/{g['tot_blk']}) "
        f"| {pct(z6['ok_blk'],z6['tot_blk'])} ({z6['ok_blk']}/{z6['tot_blk']}) "
        f"| {pct(z9['ok_blk'],z9['tot_blk'])} ({z9['ok_blk']}/{z9['tot_blk']}) "
        f"| {diff_exec:+.2f}% |"
    )
    
    lines.append(
        f"| **PRM 挽救题数** | - | - | {z6['salvaged']} 题 | {z9['salvaged']} 题 | |"
    )

    lines.append("\n## 二、 分学科（Subject）准确率对比\n")
    lines.append("| 学科 Type | SFT Baseline | GRPO-1800 | ZPPO-600 | ZPPO-960 |")
    lines.append("|---|---|---|---|---|")
    
    unique_types = sorted(list(z9["by_type"].keys()))
    for t in unique_types:
        sc = s["by_type"].get(t, {"correct": 0, "total": 1})
        gc_ = g["by_type"].get(t, {"correct": 0, "total": 1})
        z6c = z6["by_type"].get(t, {"correct": 0, "total": 1})
        z9c = z9["by_type"].get(t, {"correct": 0, "total": 1})
        lines.append(
            f"| {t} | {pct(sc['correct'], sc['total'])} ({sc['correct']}/{sc['total']}) "
            f"| {pct(gc_['correct'], gc_['total'])} ({gc_['correct']}/{gc_['total']}) "

            f"| {pct(z6c['correct'], z6c['total'])} ({z6c['correct']}/{z6c['total']}) "
            f"| {pct(z9c['correct'], z9c['total'])} ({z9c['correct']}/{z9c['total']}) |"
        )

    report_out = os.path.join(SCRATCH, "hard_math_benchmark_report.md")
    user_facing_out = r"C:\Users\rick john\.gemini\antigravity\brain\8ceedcb6-148d-478c-b186-c0bb494fe889\hard_math_benchmark_report.md"
    
    report_text = "\n".join(lines)
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(user_facing_out, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"Report compiled successfully at: {user_facing_out}")

def main():
    with open(TEST_SET, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} hard test questions.")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    def load_tokenizer():
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tok.pad_token     = tok.eos_token
        tok.padding_side  = "left"
        return tok

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
        )

    # 1. Evaluate SFT
    sft_out = os.path.join(SCRATCH, "bench250_hard_sft.json")
    if not os.path.exists(sft_out) or len(json.load(open(sft_out, encoding="utf-8"))) < len(questions):
        tok = load_tokenizer()
        model = load_base()
        evaluate(model, tok, questions, sft_out, "SFT")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 2. Evaluate GRPO-1800
    grpo_out = os.path.join(SCRATCH, "bench250_hard_grpo1800.json")
    if not os.path.exists(grpo_out) or len(json.load(open(grpo_out, encoding="utf-8"))) < len(questions):
        tok = load_tokenizer()
        base = load_base()
        model = PeftModel.from_pretrained(base, GRPO_ADAPTER)
        evaluate(model, tok, questions, grpo_out, "GRPO-1800")
        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Evaluate ZPPO-600
    z600_out = os.path.join(SCRATCH, "bench250_hard_zppo600.json")
    if not os.path.exists(z600_out) or len(json.load(open(z600_out, encoding="utf-8"))) < len(questions):
        tok = load_tokenizer()
        base = load_base()
        model = PeftModel.from_pretrained(base, ZPPO_600_DIR)
        evaluate(model, tok, questions, z600_out, "ZPPO-600")
        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Evaluate ZPPO-960
    z960_out = os.path.join(SCRATCH, "bench250_hard_zppo960.json")
    if not os.path.exists(z960_out) or len(json.load(open(z960_out, encoding="utf-8"))) < len(questions):
        tok = load_tokenizer()
        base = load_base()
        model = PeftModel.from_pretrained(base, ZPPO_960_DIR)
        evaluate(model, tok, questions, z960_out, "ZPPO-960")
        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    # 5. Compile report
    print("\nGenerating final comparative report on Hard Math Subset...")
    sft_data  = json.load(open(sft_out, encoding="utf-8"))
    grpo_data = json.load(open(grpo_out, encoding="utf-8"))
    z600_data = json.load(open(z600_out, encoding="utf-8"))
    z960_data = json.load(open(z960_out, encoding="utf-8"))
    make_final_report(sft_data, grpo_data, z600_data, z960_data)

if __name__ == "__main__":
    main()
