"""
Comparative 500-Question Benchmark (ZPPO-960 vs ZPPO-600)
- Reads cached SFT and GRPO-1800 results from scratch
- Evaluates ZPPO-960 on 500 questions -> bench500_zppo960.json
- Evaluates ZPPO-600 on 500 questions -> bench500_zppo600.json
- Compiles a comparative report comparing SFT vs GRPO-1800 vs ZPPO-600 vs ZPPO-960
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, re, gc, json, time
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
ZPPO_960_DIR = r"E:\math workspace\grpo_output_discounted\zppo_checkpoint\checkpoint_step960"
ZPPO_600_DIR = r"E:\math workspace\grpo_output_discounted\zppo_checkpoint\checkpoint_step600"
GRPO_ADAPTER = r"E:\math workspace\grpo_output_discounted\checkpoint_step1800_v3"

SCRATCH = r"C:\Users\rick john\.gemini\antigravity\brain\8ceedcb6-148d-478c-b186-c0bb494fe889\scratch"
TEST_SET = os.path.join(SCRATCH, "numina_500_test.json")

SFT_SOURCE  = os.path.join(SCRATCH, "bench500_sft.json")
GRPO_SOURCE = os.path.join(SCRATCH, "bench500_grpo1800.json")

ZPPO_960_OUT = os.path.join(SCRATCH, "bench500_zppo960.json")
ZPPO_600_OUT = os.path.join(SCRATCH, "bench500_zppo600.json")
REPORT_PATH  = r"C:\Users\rick john\.gemini\antigravity\brain\8ceedcb6-148d-478c-b186-c0bb494fe889\bench500_overfitting_report.md"

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
        print(f"\n  [{model_name}] Q{idx+1}/500 | Target: {target[:40]}")

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

                is_finished = False
                if any(x in assistant_txt for x in ["boxed", "The answer is", "Final Answer"]):
                    is_finished = True
                
                last_token = gen_ids[-1].item()
                if last_token in eos_ids:
                    is_finished = True
                
                if not has_code:
                    break


        # Compile final trajectory text
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
    sc_opp, sc_ok = 0, 0
    for x in data:
        turns = x["turns"]
        if len(turns) > 1 and turns[0]["has_code"] and not turns[0]["executed_success"]:
            sc_opp += 1
            if any(t["executed_success"] for t in turns[1:]):
                sc_ok += 1
    return {
        "total":    total,
        "correct":  correct,
        "code_q":   code_q,
        "tot_blk":  tot_blk,
        "ok_blk":   ok_blk,
        "sc_opp":   sc_opp,
        "sc_ok":    sc_ok,
    }

def make_report(questions, sft_data, grpo_data, zppo600_data, zppo960_data):
    s  = get_stats(sft_data)
    g  = get_stats(grpo_data)
    z6 = get_stats(zppo600_data)
    z9 = get_stats(zppo960_data)

    def pct(a, b): return f"{a/b*100:.1f}%" if b else "N/A"

    lines = []
    lines.append("# SFT vs GRPO-1800 vs ZPPO-600 vs ZPPO-960  |  500 题 Overfitting 对比报告\n")
    lines.append(f"> 测试集：NuminaMath 500 题（未参与训练）| 解码：贪婪 | 沙箱：6 轮 × 1024 token/轮 | 挽救阈值：≥ 0.40\n")
    lines.append("## 核心数据对比\n")
    lines.append("| 指标 | SFT Baseline | GRPO-1800 | ZPPO-600 (中期) | ZPPO-960 (后期) | 性能变化 (960 vs 600) |")
    lines.append("|---|---|---|---|---|---|")
    
    diff_val = (z9['correct']/z9['total'] - z6['correct']/z6['total']) * 100
    lines.append(
        f"| **准确率 Accuracy** | {pct(s['correct'],s['total'])} ({s['correct']}/{s['total']}) "
        f"| {pct(g['correct'],g['total'])} ({g['correct']}/{g['total']}) "
        f"| {pct(z6['correct'],z6['total'])} ({z6['correct']}/{z6['total']}) "
        f"| {pct(z9['correct'],z9['total'])} ({z9['correct']}/{z9['total']}) "
        f"| {diff_val:+.1f}% |"
    )
    
    diff_code = (z9['code_q']/z9['total'] - z6['code_q']/z6['total']) * 100
    lines.append(
        f"| **写码率 Code Gen** | {pct(s['code_q'],s['total'])} "
        f"| {pct(g['code_q'],g['total'])} "
        f"| {pct(z6['code_q'],z6['total'])} "
        f"| {pct(z9['code_q'],z9['total'])} "
        f"| {diff_code:+.1f}% |"
    )
    
    diff_exec = (z9['ok_blk']/z9['tot_blk'] - z6['ok_blk']/z6['tot_blk']) * 100 if z6['tot_blk'] and z9['tot_blk'] else 0.0
    lines.append(
        f"| **代码执行成功率** | {pct(s['ok_blk'],s['tot_blk'])} ({s['ok_blk']}/{s['tot_blk']}) "
        f"| {pct(g['ok_blk'],g['tot_blk'])} ({g['ok_blk']}/{g['tot_blk']}) "
        f"| {pct(z6['ok_blk'],z6['tot_blk'])} ({z6['ok_blk']}/{z6['tot_blk']}) "
        f"| {pct(z9['ok_blk'],z9['tot_blk'])} ({z9['ok_blk']}/{z9['tot_blk']}) "
        f"| {diff_exec:+.1f}% |"
    )
    
    sc_s_pct = pct(s['sc_ok'], s['sc_opp'])
    sc_g_pct = pct(g['sc_ok'], g['sc_opp'])
    sc_z6_pct = pct(z6['sc_ok'], z6['sc_opp'])
    sc_z9_pct = pct(z9['sc_ok'], z9['sc_opp'])
    diff_sc = (z9['sc_ok']/z9['sc_opp'] - z6['sc_ok']/z6['sc_opp']) * 100 if z6['sc_opp'] and z9['sc_opp'] else 0.0
    lines.append(
        f"| **首轮纠错率** | {sc_s_pct} ({s['sc_ok']}/{s['sc_opp']}) "
        f"| {sc_g_pct} ({g['sc_ok']}/{g['sc_opp']}) "
        f"| {sc_z6_pct} ({z6['sc_ok']}/{z6['sc_opp']}) "
        f"| {sc_z9_pct} ({z9['sc_ok']}/{z9['sc_opp']}) "
        f"| {diff_sc:+.1f}% |"
    )

    lines.append("\n## 逐题对比 (前50题展示)\n")
    lines.append("| # | 题目(前50字) | 答案 | SFT | GRPO | ZPPO-600 | ZPPO-960 |")
    lines.append("|---|---|---|---|---|---|---|")
    for i in range(min(50, len(questions))):
        q = questions[i]
        sr = sft_data[i]
        gr = grpo_data[i]
        z6r = zppo600_data[i]
        z9r = zppo960_data[i]
        qt = q["prompt"].replace("\n", " ").replace("|", "I")[:50]
        lines.append(
            f"| {i+1} | {qt}... | `{q['answer']}` "
            f"| {'✅' if sr['matched'] else '❌'} "
            f"| {'✅' if gr['matched'] else '❌'} "
            f"| {'✅' if z6r['matched'] else '❌'} "
            f"| {'✅' if z9r['matched'] else '❌'} |"
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {REPORT_PATH}")

def main():
    with open(TEST_SET, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} test questions.")

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
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
        )
        return model

    def free(model):
        del model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

    # ── Phase 1 & 2: Load SFT and GRPO from cached 500-question benchmarks ──
    with open(SFT_SOURCE, "r", encoding="utf-8") as f:
        sft_data = json.load(f)
    with open(GRPO_SOURCE, "r", encoding="utf-8") as f:
        grpo_data = json.load(f)

    # ── Phase 3: ZPPO-960 500-evaluation ──────────────────────────────────────
    z960_done = os.path.exists(ZPPO_960_OUT) and len(json.load(open(ZPPO_960_OUT, encoding="utf-8"))) == len(questions)
    if not z960_done:
        print("\n[Phase 3] Evaluating ZPPO-960...")
        tok = load_tokenizer()
        base = load_base()
        model = PeftModel.from_pretrained(base, ZPPO_960_DIR)
        evaluate(model, tok, questions, ZPPO_960_OUT, "ZPPO-960")
        free(model)
    else:
        print("[Phase 3] ZPPO-960 already evaluated, skipping.")

    # ── Phase 4: ZPPO-600 500-evaluation ──────────────────────────────────────
    z600_done = os.path.exists(ZPPO_600_OUT) and len(json.load(open(ZPPO_600_OUT, encoding="utf-8"))) == len(questions)
    if not z600_done:
        print("\n[Phase 4] Evaluating ZPPO-600...")
        tok = load_tokenizer()
        base = load_base()
        model = PeftModel.from_pretrained(base, ZPPO_600_DIR)
        evaluate(model, tok, questions, ZPPO_600_OUT, "ZPPO-600")
        free(model)
    else:
        print("[Phase 4] ZPPO-600 already evaluated, skipping.")

    # ── Phase 5: Compile Unified Overfitting Report ───────────────────────────
    print("\n[Phase 5] Generating overfitting comparison report...")
    zppo960_data = json.load(open(ZPPO_960_OUT, encoding="utf-8"))
    zppo600_data = json.load(open(ZPPO_600_OUT, encoding="utf-8"))
    make_report(questions, sft_data, grpo_data, zppo600_data, zppo960_data)

if __name__ == "__main__":
    main()
