"""
DPO 数据对生成器：从 GRPO Stage 2 训练日志中提取重复 vs 干净的偏好数据对。

数据来源: training_log_v2.txt (Steps 161 ~ latest)
输出格式: JSONL, 每行 {"prompt": ..., "chosen": ..., "rejected": ...}

两种配对方法:
  方法1 (跨候选配对): 同一 Step 中, chosen=干净且正确, rejected=重复且正确
  方法2 (自编辑截断): 对重复候选, chosen=程序化截断版, rejected=原始重复版

运行方式:
  python generate_dpo_pairs.py [--max-step 240] [--output dpo_pairs.jsonl]

注: 此脚本只生成数据，不执行训练。
"""
import re
import json
import argparse
import os
import sys
import io

# Resolve repo root from this file's location so core/ is importable
# regardless of the current working directory.
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.evaluator import compare_math_answers

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# 1. 重复检测
# ─────────────────────────────────────────────────────────────

def is_candidate_for_repetition_check(stripped):
    """
    判断该行是否需要进行重复性检测。
    条件：
      - 字符长度 > 2
      - 或者只包含重复符号（如 '.', '-', '*', '_', '+', '=', '#', '~'）
    """
    if len(stripped) > 2:
        return True
    if re.match(r'^[-.*_+=#~]+$', stripped):
        return True
    return False


def compute_repetition_stats(text):
    """
    计算文本的重复统计指标。
    返回 dict: max_streak, unique_ratio, total_lines, unique_lines, fg_ratio
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return {"max_streak": 0, "unique_ratio": 1.0, "total_lines": 0, "unique_lines": 0, "fg_ratio": 1.0}

    # 最长连续相同行
    max_streak = 1
    current_streak = 1
    for i in range(1, len(lines)):
        if lines[i] == lines[i - 1] and is_candidate_for_repetition_check(lines[i]):
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1

    unique_ratio = len(set(lines)) / len(lines)

    # 5-gram word unique ratio (用于检测语义/结构性重复)
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) >= 5:
        ngrams = [tuple(words[i:i+5]) for i in range(len(words) - 5 + 1)]
        fg_ratio = len(set(ngrams)) / len(ngrams)
    else:
        fg_ratio = 1.0

    return {
        "max_streak": max_streak,
        "unique_ratio": unique_ratio,
        "total_lines": len(lines),
        "unique_lines": len(set(lines)),
        "fg_ratio": fg_ratio,
    }


def is_severely_repetitive(stats):
    """判断是否为严重重复：连续 >=3行 相同，或独特行率 <50%，或语义/结构性重复"""
    if stats["max_streak"] >= 3:
        return True
    if stats["unique_ratio"] < 0.50:
        return True
    # 语义重复检测：行数较多且独特行比例低于 85%
    if stats["total_lines"] > 40 and stats["unique_ratio"] < 0.85:
        return True
    # 5-gram 独特性低于 75%
    if stats["fg_ratio"] < 0.75:
        return True
    return False


def is_clean(stats):
    """判断是否为干净输出：连续重复 <=2 且独特行率 >=85% 且 5-gram 独特率 >=80%"""
    return stats["max_streak"] <= 2 and stats["unique_ratio"] >= 0.85 and stats["fg_ratio"] >= 0.80



# ─────────────────────────────────────────────────────────────
# 2. 重复截断 (方法2的核心)
# ─────────────────────────────────────────────────────────────

def truncate_repetition(text):
    """
    检测并截断文本中的重复部分。
    策略:
      - 忽略空行，提取所有非空行的索引和内容。
      - 检测连续重复行 (streak >= 2, 长度 > 8) 并标记删除。
      - 检测周期性重复 (period in [2,3,4], repeats >= 3, 至少有一个长度 > 8) 并标记删除。
      - 重建文本时跳过被删除的行，并压缩连续空行为单个空行。
      - 返回 (truncated_text, was_modified, n_lines_removed)
    """
    lines = text.split("\n")
    non_empty = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            non_empty.append((idx, stripped))

    if not non_empty:
        return text, False, 0

    to_remove_indices = set()

    # 1. 检测连续重复行 (忽略空行)
    i = 0
    while i < len(non_empty):
        idx, stripped = non_empty[i]
        if is_candidate_for_repetition_check(stripped):
            streak_end = i + 1
            while streak_end < len(non_empty) and non_empty[streak_end][1] == stripped:
                streak_end += 1
            
            streak_len = streak_end - i
            if streak_len >= 2:
                for j in range(i + 1, streak_end):
                    to_remove_indices.add(non_empty[j][0])
                i = streak_end
                continue
        i += 1

    # 2. 检测周期性重复行 (忽略空行)
    i = 0
    while i < len(non_empty):
        if i + 4 < len(non_empty):
            found_period = False
            for period in [2, 3, 4]:
                if i + period * 3 <= len(non_empty):
                    pattern = [non_empty[j][1] for j in range(i, i + period)]
                    if any(is_candidate_for_repetition_check(p) for p in pattern):
                        repeats = 1
                        pos = i + period
                        while pos + period <= len(non_empty):
                            chunk = [non_empty[j][1] for j in range(pos, pos + period)]
                            if chunk == pattern:
                                repeats += 1
                                pos += period
                            else:
                                break
                        if repeats >= 3:
                            for j in range(i + period, pos):
                                to_remove_indices.add(non_empty[j][0])
                            i = pos
                            found_period = True
                            break
            if found_period:
                continue
        i += 1

    if not to_remove_indices:
        return text, False, 0

    # 重建文本并压缩连续空行
    new_lines = []
    for idx, line in enumerate(lines):
        if idx not in to_remove_indices:
            new_lines.append(line)

    final_lines = []
    for line in new_lines:
        if not line.strip():
            if not final_lines or final_lines[-1].strip():
                final_lines.append("")
        else:
            final_lines.append(line)

    # 去除首尾空行和无意义的符号行
    while final_lines:
        first_line = final_lines[0].strip()
        if not first_line or re.match(r'^[-.*_+=#~]+$', first_line) or first_line == "...":
            final_lines.pop(0)
        else:
            break

    while final_lines:
        last_line = final_lines[-1].strip()
        if not last_line or re.match(r'^[-.*_+=#~]+$', last_line) or last_line == "...":
            final_lines.pop()
        else:
            break

    truncated_text = "\n".join(final_lines)
    n_removed = len(lines) - len(final_lines)
    return truncated_text, True, n_removed


# ─────────────────────────────────────────────────────────────
# 3. 日志解析
# ─────────────────────────────────────────────────────────────

def parse_training_log(log_path, min_step=161, max_step=300):
    """
    解析 training_log_v2.txt, 返回 step 数据列表（保留重复/重启的步骤作为独立样本）。
    每个 step: {step, prompt, target, rewards: [...], candidates: [...]}
    """
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 按顺序收集每个 step 的文本块（不使用 dict 去重，以保留重启后的新老数据）
    step_blocks = []
    current_step = None
    current_block = []
    
    for line in lines:
        m = re.match(r"^--- Step (\d+) ---", line)
        if m:
            if current_step is not None:
                step_blocks.append((current_step, "".join(current_block)))
            current_step = int(m.group(1))
            current_block = []
            continue
        if current_step is not None:
            current_block.append(line)
            
    if current_step is not None:
        step_blocks.append((current_step, "".join(current_block)))

    results = []
    for step_num, block in step_blocks:
        if step_num < min_step or step_num > max_step:
            continue

        # 提取 prompt
        q_match = re.search(r"Question: (.+?)(?=\nTarget Answer:)", block, re.DOTALL)
        if not q_match:
            continue
        prompt = q_match.group(1).strip()

        # 提取 target
        t_match = re.search(r"Target Answer:\s*(.+)", block)
        target = t_match.group(1).strip() if t_match else ""

        # 提取 rewards
        r_match = re.search(r"Rewards: \[([^\]]+)\]", block)
        if not r_match:
            continue
        rewards = [float(x.strip()) for x in r_match.group(1).split(",")]

        # 提取各候选文本
        cand_splits = re.split(r"--- Candidate \d+ \(Reward: [-\d.]+\) ---", block)
        candidate_texts = []
        for c in cand_splits[1:]:
            c_clean = re.split(r"\s*Loss:\s*[-\d.]+\s*\|\s*Mean Reward:", c)[0]
            candidate_texts.append(c_clean.strip())

        if len(candidate_texts) != len(rewards):
            n = min(len(candidate_texts), len(rewards))
            candidate_texts = candidate_texts[:n]
            rewards = rewards[:n]

        results.append({
            "step": step_num,
            "prompt": prompt,
            "target": target,
            "rewards": rewards,
            "candidates": candidate_texts,
        })

    return results


# ─────────────────────────────────────────────────────────────
# 4. DPO 数据对生成
# ─────────────────────────────────────────────────────────────

def extract_final_stdout(text):
    """
    从候选文本中提取最后一个 Observation: 块的内容（即 Python 执行的输出）。
    """
    matches = re.findall(r'Observation:\s*(.*?)(?=\n\n|\n---|\n<STEP>|\n$|$)', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""


def is_mathematically_correct(text, target):
    """
    判断候选输出是否在数学上正确（与 evaluator_v2 的判断逻辑一致）。
    1. 提取 Python 执行输出 (stdout)，使用 compare_math_answers 进行精确匹配。
    2. 如果没有执行输出，或者精确匹配失败，使用文本子串和单位容忍做兜底校验。
    """
    if not target:
        return True
        
    # 1. 提取代码执行输出并进行判定
    stdout = extract_final_stdout(text)
    if stdout:
        if compare_math_answers(stdout, target):
            return True
            
    # 2. 兜底校验：文本中包含答案（防粘连处理）
    text_lower = text.lower()
    norm_target = target.lower().replace(".0", "")
    
    if norm_target in text_lower:
        return True
        
    if norm_target.isdigit():
        if f" {norm_target} " in f" {text_lower} ":
            return True
        if f"={norm_target}" in text_lower.replace(" ", ""):
            return True
        if f"${norm_target}" in text_lower.replace(" ", "") or f"{norm_target}%" in text_lower.replace(" ", ""):
            return True
            
    return False


def generate_dpo_pairs(step_data_list):
    """
    从 step 数据生成 DPO 偏好对。
    返回: (method1_pairs, method2_pairs)
    每个 pair: {"prompt": str, "chosen": str, "rejected": str, "source": str, "step": int, "info": str}
    """
    method1_pairs = []
    method2_pairs = []

    for step_data in step_data_list:
        prompt = step_data["prompt"]
        rewards = step_data["rewards"]
        candidates = step_data["candidates"]
        step_num = step_data["step"]
        target = step_data["target"]

        # 为每个候选计算重复统计
        cand_stats = []
        for i, cand_text in enumerate(candidates):
            stats = compute_repetition_stats(cand_text)
            stats["index"] = i
            stats["reward"] = rewards[i] if i < len(rewards) else 0.0
            stats["text"] = cand_text
            cand_stats.append(stats)

        # ── 方法1: 跨候选配对 ──
        # 找出: 正奖励 + 干净 + 包含正确答案 的 候选 (chosen)
        # 找出: 正奖励 + 重复 的 候选 (rejected)
        clean_positive = [
            s for s in cand_stats 
            if s["reward"] > 0 and is_clean(s) and is_mathematically_correct(s["text"], target)
        ]
        repetitive_positive = [
            s for s in cand_stats 
            if s["reward"] > 0 and is_severely_repetitive(s)
        ]

        # 如果两者都存在, 取最佳 chosen 和最差 rejected
        if clean_positive and repetitive_positive:
            # chosen: 最干净的正候选 (最高 unique_ratio)
            best_clean = max(clean_positive, key=lambda s: s["unique_ratio"])
            # rejected: 最重复的正候选 (最低 unique_ratio)
            worst_rep = min(repetitive_positive, key=lambda s: s["unique_ratio"])

            method1_pairs.append({
                "prompt": prompt,
                "chosen": best_clean["text"],
                "rejected": worst_rep["text"],
                "source": "cross_candidate",
                "step": step_num,
                "info": (
                    f"chosen: C{best_clean['index']+1} reward={best_clean['reward']:.3f} "
                    f"unique={best_clean['unique_ratio']:.2f} | "
                    f"rejected: C{worst_rep['index']+1} reward={worst_rep['reward']:.3f} "
                    f"unique={worst_rep['unique_ratio']:.2f}"
                ),
            })

        # ── 方法2: 自编辑截断 ──
        # 对每个正奖励 + 严重重复的候选, 生成截断版作为 chosen
        for s in cand_stats:
            if s["reward"] <= 0:
                continue  # 必须是算对的候选，才作为 DPO 示范
            if not is_severely_repetitive(s):
                continue
            if s["total_lines"] < 5:
                continue  # 太短, 不处理

            truncated, was_modified, n_removed = truncate_repetition(s["text"])
            if not was_modified or n_removed < 2:
                continue  # 截断效果不明显

            # 确保截断后文本比原始短至少20%
            orig_len = len(s["text"])
            trunc_len = len(truncated)
            if trunc_len >= orig_len * 0.85:
                continue  # 差异太小, 不构成有效偏好对

            # 确保截断后文本不为空且有实质内容
            if len(truncated.strip()) < 50:
                continue

            # 确保截断后的 Chosen 依然包含正确答案！
            if not is_mathematically_correct(truncated, target):
                continue

            method2_pairs.append({
                "prompt": prompt,
                "chosen": truncated,
                "rejected": s["text"],
                "source": "self_edit",
                "step": step_num,
                "info": (
                    f"C{s['index']+1} reward={s['reward']:.3f} | "
                    f"original: {s['total_lines']} lines, unique={s['unique_ratio']:.2f}, "
                    f"streak={s['max_streak']} | "
                    f"truncated: removed {n_removed} lines ({n_removed/s['total_lines']*100:.0f}%)"
                ),
            })

    return method1_pairs, method2_pairs


# ─────────────────────────────────────────────────────────────
# 5. 去重与质量筛选
# ─────────────────────────────────────────────────────────────

def deduplicate_pairs(pairs):
    """
    去重: 如果同一个 Step 同时产生了方法1和方法2的 pair,
    且方法2的 rejected 文本与方法1的 rejected 相同, 则保留方法1 (质量更高)。
    """
    seen_rejected = set()
    unique_pairs = []
    for p in pairs:
        key = (p["step"], p["rejected"][:200])  # 用前200字符作为近似 key
        if key not in seen_rejected:
            seen_rejected.add(key)
            unique_pairs.append(p)
    return unique_pairs


def clean_chosen_text(text):
    """
    清理 Chosen 文本尾部悬挂的 Suspicious/重复的模板化句式（如 and it's all confirmed... 等）。
    """
    lines = text.split("\n")
    while lines:
        last_line = lines[-1].strip()
        is_suspect = False
        if not last_line or re.match(r'^[-.*_+=#~]+$', last_line) or last_line == "...":
            is_suspect = True
        elif any(phrase in last_line.lower() for phrase in [
            "confirmed by the python code",
            "verified by the code",
            "that's that",
            "let me know if you need any clarification",
            "let me know if you have any other questions",
            "you're doing great",
            "you've been a great assistant",
        ]):
            is_suspect = True
            
        if is_suspect:
            lines.pop()
        else:
            break
    return "\n".join(lines)


def quality_filter(pairs):
    """
    质量过滤:
    1. chosen 和 rejected 不能完全相同
    2. chosen 的重复统计必须优于 rejected
    3. chosen 必须有实质内容 (>=50 字符)
    """
    filtered = []
    for p in pairs:
        # 清理 chosen 文本的尾部垃圾与模板语句
        p["chosen"] = clean_chosen_text(p["chosen"])

        # 不能相同
        if p["chosen"].strip() == p["rejected"].strip():
            continue

        # chosen 必须比 rejected 更干净
        chosen_stats = compute_repetition_stats(p["chosen"])
        rejected_stats = compute_repetition_stats(p["rejected"])

        if chosen_stats["unique_ratio"] <= rejected_stats["unique_ratio"]:
            continue  # chosen 居然比 rejected 更重复?

        # chosen 必须有内容
        if len(p["chosen"].strip()) < 50:
            continue

        p["chosen_unique_ratio"] = chosen_stats["unique_ratio"]
        p["rejected_unique_ratio"] = rejected_stats["unique_ratio"]
        filtered.append(p)

    return filtered


# ─────────────────────────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs from GRPO training log")
    parser.add_argument("--log-path", default=str(REPO_ROOT / "grpo_output_discounted" / "training_log_v2.txt"),
                        help="Path to training_log_v2.txt")
    parser.add_argument("--min-step", type=int, default=1, help="Minimum step to include")
    parser.add_argument("--max-step", type=int, default=300, help="Maximum step to include")
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "dpo_anti_repetition.jsonl"),
                        help="Output JSONL path")
    args = parser.parse_args()

    print(f"Parsing training log: {args.log_path}")
    print(f"Step range: {args.min_step} ~ {args.max_step}")

    # 支持同时加载 Stage 1 和 Stage 2 的日志来扩大数据集
    step_data = []
    if os.path.exists(args.log_path):
        step_data += parse_training_log(args.log_path, args.min_step, args.max_step)
        
    stage1_log = args.log_path.replace("training_log_v2.txt", "training_log.txt")
    if args.min_step <= 160 and os.path.exists(stage1_log) and stage1_log != args.log_path:
        print(f"Also parsing Stage 1 log: {stage1_log}")
        step_data += parse_training_log(stage1_log, args.min_step, min(160, args.max_step))

    # 按 Step 号升序排列
    step_data.sort(key=lambda x: x["step"])
    print(f"Parsed {len(step_data)} steps total\n")

    # 统计重复概况
    total_candidates = 0
    severe_candidates = 0
    for sd in step_data:
        for i, cand in enumerate(sd["candidates"]):
            stats = compute_repetition_stats(cand)
            total_candidates += 1
            if is_severely_repetitive(stats):
                severe_candidates += 1
    print(f"Total candidates: {total_candidates}")
    print(f"Severely repetitive: {severe_candidates} ({severe_candidates/max(total_candidates,1)*100:.1f}%)\n")

    # 生成配对
    m1_pairs, m2_pairs = generate_dpo_pairs(step_data)
    print(f"Method 1 (cross-candidate) raw pairs: {len(m1_pairs)}")
    print(f"Method 2 (self-edit) raw pairs:        {len(m2_pairs)}")

    # 合并, 方法1优先
    all_pairs = m1_pairs + m2_pairs
    all_pairs = deduplicate_pairs(all_pairs)
    print(f"After dedup: {len(all_pairs)}")

    all_pairs = quality_filter(all_pairs)
    print(f"After quality filter: {len(all_pairs)}")

    # 输出统计
    m1_final = [p for p in all_pairs if p["source"] == "cross_candidate"]
    m2_final = [p for p in all_pairs if p["source"] == "self_edit"]
    print(f"\nFinal breakdown:")
    print(f"  Cross-candidate pairs: {len(m1_final)}")
    print(f"  Self-edit pairs:       {len(m2_final)}")
    print(f"  Total:                 {len(all_pairs)}")

    # 输出详情
    print(f"\n{'=' * 90}")
    print("PAIR DETAILS")
    print(f"{'=' * 90}")
    for i, p in enumerate(all_pairs):
        chosen_preview = p["chosen"][:80].replace("\n", " ")
        rejected_preview = p["rejected"][:80].replace("\n", " ")
        print(f"\n  Pair {i+1} [Step {p['step']}] ({p['source']})")
        print(f"    {p['info']}")
        print(f"    Chosen  ({len(p['chosen']):>5} chars, uniq={p.get('chosen_unique_ratio', 0):.2f}): {chosen_preview}...")
        print(f"    Rejected({len(p['rejected']):>5} chars, uniq={p.get('rejected_unique_ratio', 0):.2f}): {rejected_preview}...")

    # 写入 JSONL
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in all_pairs:
            record = {
                "prompt": p["prompt"],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
                "source": p["source"],
                "step": p["step"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 90}")
    print(f"Output written to: {args.output}")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
