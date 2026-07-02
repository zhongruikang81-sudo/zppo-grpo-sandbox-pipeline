import ast
import sympy as sp
import re
import os
import json
from typing import Tuple, List
from sandbox import clean_and_dedent, execute_code

# Reward and Penalty weight settings
STEP_SUCCESS_REWARD = 0.1
STEP_FAIL_REWARD = -0.1
MAX_CORRECTNESS_REWARD = 0.9


def clean_str(s: str) -> str:
    return s.strip().replace(" ", "").lower()

def compare_math_answers(output_str: str, target_str: str) -> bool:
    """
    Deterministically compares execution output against the target mathematical answer.
    Supports exact string match, numerical float match, and symbolic algebraic match.
    Does NOT allow unit mismatches (powers of 10) in rule matching.
    """
    out = output_str.strip()
    target = target_str.strip()
    
    if not out or not target:
        return False
        
    # Helper to clean percent symbols
    def clean_percent(s: str) -> str:
        s = s.strip()
        if s.endswith("%"):
            s = s[:-1].strip()
        return s
        
    out_clean = clean_percent(out)
    target_clean = clean_percent(target)
    
    # 1. Direct string match after basic cleanup
    if clean_str(out_clean) == clean_str(target_clean):
        return True
        
    # Check the last printed line as the final result
    last_line = out.split("\n")[-1].strip()
    last_line_clean = clean_percent(last_line)
    if clean_str(last_line_clean) == clean_str(target_clean):
        return True

    # Helper function to check if two floats are close
    def is_close_without_scaling(val1: float, val2: float) -> bool:
        if val1 == 0.0 and val2 == 0.0:
            return True
        if val1 == 0.0 or val2 == 0.0:
            return False
        # Check standard relative error (within 1%)
        rel_err = abs(val1 - val2) / max(abs(val1), abs(val2))
        return rel_err < 0.01
        
    # 2. Float equivalence comparison
    try:
        val_out = float(last_line_clean)
        val_target = float(target_clean)
        if is_close_without_scaling(val_out, val_target):
            return True
    except ValueError:
        pass

    # 2b. Regex number extraction from last line
    try:
        nums = re.findall(r'-?\d+\.?\d*', last_line_clean)
        if nums:
            val_out = float(nums[-1])
            val_target = float(target_clean)
            if is_close_without_scaling(val_out, val_target):
                return True
    except (ValueError, TypeError):
        pass
        
    # Preprocess strings to remove common namespace prefixes (e.g. sp.sin -> sin)
    def clean_math_expr(s):
        return s.replace("sp.", "").replace("sympy.", "").replace("np.", "").replace("numpy.", "")
        
    # 3. SymPy symbolic and numerical comparison
    try:
        expr_out = sp.sympify(clean_math_expr(last_line_clean))
        expr_target = sp.sympify(clean_math_expr(target_clean))
        
        # Check symbolic simplification
        if sp.simplify(expr_out - expr_target) == 0:
            return True
            
        # Check numerical equivalence for numeric expressions
        if expr_out.is_number and expr_target.is_number:
            val_out = float(expr_out.evalf())
            val_target = float(expr_target.evalf())
            if is_close_without_scaling(val_out, val_target):
                return True
    except Exception:
        pass
        
    return False

SYSTEM_PROMPT = """你是一个极其严谨且善于鼓励学生的数学与代码评估专家。
请仔细比对【原始问题】、【标准正确答案】和【学生答题轨迹】，按照以下阶梯标准给学生打分（salvage score）：

【打分阶梯标准】
- 如果学生完全偏离题意、逻辑混乱、或者写了与问题无关的敷衍/万金油代码（如直接 print 一个常数，或只打印 'Some completion'），给出 <score>0.0</score>。
- 如果学生做做出【合理尝试】：正确理解了题意，定义了与题目相关的变量，且代码能够成功执行，但解题路径不完整或公式有明显偏差，给出 <score>0.15</score>。
- 如果学生做出了【大体正确的推导】：解题路径和代码框架基本完整且正确，但由于细节公式写错或边界条件漏掉导致结果错误（例如计算百分比时在代码里乘以了两次100导致结果扩大100倍），给出 <score>0.30</score>。
- 如果学生做出了【完美推导但有极小笔误】：
  - 逻辑、思路和代码基本 100% 正确，仅因为变量名拼写、缩进、浮点数取舍等极小笔误导致报错或结果微小偏差，给出 <score>0.40</score>。
- 如果学生做出了【完全正确的计算，仅由于单位换算/表达格式/百分比与小数形式不同】：
  - **学生计算和推导完全正确，仅仅因为单位/数量级表达（如结果是 703.7 m，而标准答案是 70336.0 cm；或者百分比写成小数 0.333 而标准答案是 33.33），或者输出了百分号（如 33.33%），这属于完全正确的解答，应当给予满分 <score>0.50</score>**。

【给分核心原则（请牢记）】
- **鼓励尝试，惩罚敷衍**：只要学生做出了合理解题尝试（即：定义了相关变量、写了物理/数学计算公式并执行，而非一上来就直接 print 一个写死的常数），哪怕算错或未完成，也**必须**根据程度给出 0.15 或 0.30 分的尝试分。我们鼓励有价值的推导，杜绝一概而论给 0.0 分！

【重要效率与评估要求（请严格执行）】
1. **极简推理**：你的思维链（CoT）和分析过程必须极其精炼，严禁大篇幅讨论，直接用最少的话切入主题，快速给出结论。
2. **禁止多余猜测**：客观判定最终代码和结果。如果学生最终代码逻辑完全正确，不要过度猜测历史步骤的矛盾、环境Bug或作弊等无关因素。
3. **极简输出**：必须使用指定的 XML 标签。<reason> 标签内必须只保留 1-2 句极简短的解释，严禁长篇大论。

你必须严格按照以下 XML 标签格式输出，不要包含 ```json 等多余的 Markdown 格式：
<score>分数(0.0/0.15/0.30/0.40/0.50之一)</score>
<reason>用1-2句话极简描述打分依据</reason>
"""

def parse_ds_score(response_text: str) -> float:
    """
    Standard XML regex parser to extract salvage score from DeepSeek response.
    """
    if not response_text:
        return 0.0
    
    # 1. Main Regex match for <score>...</score>
    match = re.search(r'<score>\s*(-?[0-9.]+)\s*</score>', response_text, re.IGNORECASE)
    if match:
        try:
            score = float(match.group(1))
            # Safety checks & boundary clamping
            if 0.0 <= score <= 0.50:
                return score
            elif score > 0.50:
                return 0.50
            return 0.0
        except ValueError:
            pass
            
    # 2. Fallback regex match for bracketed [score] or raw score
    backup_match = re.search(r'score[^\d]{1,10}(-?[0-9.]+)', response_text, re.IGNORECASE)
    if backup_match:
        try:
            score = float(backup_match.group(1))
            if 0.0 <= score <= 0.50:
                return score
            elif score > 0.50:
                return 0.50
        except ValueError:
            pass
            
    print(f"[Warning] Failed to parse salvage score from DS output: {response_text[:120]}...")
    return 0.0

def get_prm_salvage_score(completion_text: str, target_answer: str, prompt_question: str = None) -> float:
    """
    Queries DeepSeek V3 API to perform reasoning audit and salvage logical correct paths.
    """
    api_key = "***REMOVED***"
    base_url = "https://api.deepseek.com"
    judge_model = "deepseek-v4-pro"
    
    import openai
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # Set default question context if missing
    question_str = prompt_question if prompt_question else "[Unavailable]"
    
    user_content = f"""【原始问题】
{question_str}

【标准正确答案】
{target_answer}

【学生答题轨迹】
{completion_text}

请给出你的评估结果（严格遵循 XML 预设格式）："""

    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=1024,
            timeout=45.0  # Allow up to 45 seconds for deep reasoning models (CoT might be long)
        )
        message = response.choices[0].message
        raw_output = message.content
        reasoning = getattr(message, "reasoning_content", None)
        
        # Log to audit file
        audit_path = "E:\\math workspace\\grpo_output_discounted\\prm_audit_log.txt"
        try:
            with open(audit_path, "a", encoding="utf-8") as audit_file:
                audit_file.write(f"=== PRM Audit Step ===\n")
                audit_file.write(f"Question:\n{question_str}\n\n")
                audit_file.write(f"Target Answer: {target_answer}\n\n")
                audit_file.write(f"Reasoning CoT:\n{reasoning}\n\n")
                audit_file.write(f"DeepSeek Content:\n{raw_output}\n")
                audit_file.write(f"======================\n\n")
        except Exception as log_err:
            print(f"[Warning] Failed to write PRM audit log: {log_err}")
            
        return parse_ds_score(raw_output)
    except Exception as e:
        try:
            audit_path = "E:\\math workspace\\grpo_output_discounted\\prm_audit_log.txt"
            with open(audit_path, "a", encoding="utf-8") as audit_file:
                audit_file.write(f"=== PRM Audit FAILURE ===\nError: {e}\n=========================\n\n")
        except:
            pass
        print(f"[Warning] C-OPRM API call failed: {e}. Fallback to 0.0 reward.")
        return 0.0

def is_suspicious_code(code_str: str) -> bool:
    """
    Checks if a Python code block is a suspicious "constant print cheat" (no variable computations,
    just print of literal constant).
    """
    try:
        # Strip comments and trailing spaces, but preserve leading indentation
        clean_lines = []
        for line in code_str.split("\n"):
            line_no_comment = line.split("#")[0]
            line_stripped = line_no_comment.rstrip()
            if line_stripped.strip():
                clean_lines.append(line_stripped)
        clean_code = "\n".join(clean_lines)
        
        tree = ast.parse(clean_code)
        
        # Real logic includes:
        # - Any binary/unary operations (BinOp, UnaryOp)
        # - Any comparisons (Compare)
        # - Any loops (For, While)
        # - Any function or class definitions (FunctionDef, ClassDef)
        # - Any imports (Import, ImportFrom)
        # - Any call to functions OTHER than print (Call)
        # - Any assignment of a non-constant value (Assign)
        has_real_logic = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.For, ast.While, 
                                 ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                has_real_logic = True
                break
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    continue
                else:
                    has_real_logic = True
                    break
            if isinstance(node, ast.Assign):
                val = node.value
                if not isinstance(val, ast.Constant):
                    if not type(val).__name__ in ['Num', 'Str', 'Bytes', 'NameConstant', 'Constant']:
                        has_real_logic = True
                        break
        return not has_real_logic
    except Exception:
        return True # treat as suspicious to be safe if parsing fails


def check_repetition_safe(text: str) -> float:
    # 1. Single character repetition (25 consecutive identical characters)
    for char in ".-_*+=#~":
        if char * 25 in text:
            return 0.2
            
    # 2. Combination character repetition (7 consecutive pattern repetitions)
    for match in re.finditer(r'(([.-_*+=#~]{2,3}))\2+', text):
        pattern = match.group(2)
        if any(c in ".-_*+=#~" for c in pattern):
            full_match = match.group(0)
            streak = len(full_match) // len(pattern)
            if streak >= 7:
                return 0.2

    # 3. Consecutive line repetition (5 consecutive identical lines of length > 15)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 5:
        max_streak = 1
        current_streak = 1
        for i in range(1, len(lines)):
            if lines[i] == lines[i-1] and len(lines[i]) > 15:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        if max_streak >= 5:
            return 0.2
            
    return 0.0

def evaluate_multi_turn_completion(completion_text: str, target_answer: str = None, prompt_question: str = None) -> Tuple[float, str]:
    """
    Evaluates a single multi-turn completion (Stage 2 version).
    Parses steps, runs them in sandbox, checks formatting violations via strip-detection,
    and applies C-OPRM salvage scoring on logical attempts.
    """
    # Auto-Close unclosed <STEP> tag at the end of the completion text
    was_unclosed = False
    if "<STEP>" in completion_text:
        last_step_open = completion_text.rfind("<STEP>")
        last_step_close = completion_text.rfind("</STEP>")
        if last_step_close < last_step_open:
            completion_text += "</STEP>"
            was_unclosed = True
            
    unclosed_penalty = -0.1 if was_unclosed else 0.0

    # 1. Format Check: Strip-Detection for ```python bypass
    # Strip valid <STEP>...</STEP> blocks and check if there are raw code blocks left
    text_without_steps = re.sub(r'<STEP>.*?</STEP>', '', completion_text, flags=re.DOTALL)
    has_bypass = "```python" in text_without_steps
    is_valid_format = not has_bypass
    format_penalty = -0.15 if has_bypass else 0.0

    # 2. Mandatory First-Step Coding Check
    # If the first turn generated by the assistant does not contain <STEP>, penalize heavily (-0.5)
    first_turn = completion_text.split("Observation:\n")[0]
    first_step_penalty = -0.5 if "<STEP>" not in first_turn else 0.0

    # 3. Extract and evaluate STEP blocks
    blocks = re.findall(r'<STEP>(.*?)</STEP>', completion_text, re.DOTALL)
    
    # Length penalty (reduced 10x to prevent rewarding empty bypasses)
    length_penalty = -0.00001 * len(completion_text)

    if not blocks:
        # Severe penalty for failing to generate any step blocks, plus length and first-step penalties
        return -0.3 + length_penalty + format_penalty + first_step_penalty, ""
        
    total_step_reward = 0.0
    errors_count = 0
    accumulated_blocks = []
    final_stdout = ""
    
    previous_failed = False
    has_self_correction = False
    
    for block in blocks:
        code = clean_and_dedent(block)
        accumulated_blocks.append(code)
        
        # AST check on current block
        ast_ok = False
        try:
            ast.parse(code)
            ast_ok = True
        except SyntaxError:
            pass
            
        # Execution check on accumulated code
        exec_ok = False
        stdout = ""
        if accumulated_blocks:
            script_lines = []
            for i, b in enumerate(accumulated_blocks):
                if i == len(accumulated_blocks) - 1 and len(accumulated_blocks) > 1:
                    script_lines.append("\nimport sys\nsys.stdout.write('---STEP-SEPARATOR---\\n')\nsys.stdout.flush()\n")
                script_lines.append(b)
                script_lines.append("\n")
            full_code = "".join(script_lines)
            success, stdout, stderr = execute_code(full_code)
            if "---STEP-SEPARATOR---" in stdout:
                parts = stdout.split("---STEP-SEPARATOR---")
                stdout = parts[-1].lstrip("\n")
            if success:
                exec_ok = True
                final_stdout = stdout
                
        if ast_ok and exec_ok:
            step_score = STEP_SUCCESS_REWARD
            if previous_failed:
                has_self_correction = True
            previous_failed = False
        else:
            step_score = STEP_FAIL_REWARD
            errors_count += 1
            previous_failed = True
            
        total_step_reward += step_score
        
    avg_step_reward = total_step_reward / len(blocks)
    
    # Self-Correction Bonus: +0.2 if the model successfully debugged a prior execution failure
    self_correction_bonus = 0.2 if has_self_correction else 0.0
    
    # 4. Correctness & C-OPRM Salvage Reward
    correctness_reward = 0.0
    discount = (0.8 ** errors_count) * (0.9 ** (len(blocks) - 1))
    
    if target_answer and target_answer.strip() and target_answer.strip().lower() not in ["", "none"]:
        # Check if the code generated is suspiciously simple/cheat code (e.g. print(constant))
        suspicious = any(is_suspicious_code(b) for b in accumulated_blocks)
        
        # Match standard answer (bypass DeepSeek ONLY if not suspicious)
        if compare_math_answers(final_stdout, target_answer) and not suspicious:
            correctness_reward = MAX_CORRECTNESS_REWARD * discount
        elif is_valid_format:
            # Audit or Salvage via C-OPRM DeepSeek API
            salvage_score = get_prm_salvage_score(completion_text, target_answer, prompt_question)
            correctness_reward = salvage_score * (MAX_CORRECTNESS_REWARD / 0.5) * discount
            
    # Check for repetition and calculate penalty
    rep_penalty = check_repetition_safe(completion_text)
    
    total_reward = avg_step_reward + correctness_reward + format_penalty + length_penalty - rep_penalty + unclosed_penalty + first_step_penalty + self_correction_bonus
    return total_reward, final_stdout

def calculate_group_rewards(completions: List[str], target_answer: str = None, prompt_question: str = None) -> List[float]:
    """
    Calculates rewards for a group of G completions (Stage 2 version).
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(completions)) as executor:
        futures = [
            executor.submit(evaluate_multi_turn_completion, comp, target_answer, prompt_question)
            for comp in completions
        ]
        results = [f.result() for f in futures]
        
    scores = [r[0] for r in results]
    stdouts = [r[1] for r in results]
        
    # If target_answer was provided, correctness/salvage rewards are already computed
    if target_answer and target_answer.strip() and target_answer.strip().lower() not in ["", "none"]:
        return scores
        
    # If no target_answer, compute consensus rewards based on majority voting
    groups = []
    for i, out in enumerate(stdouts):
        if not out or not out.strip():
            continue
        found = False
        for group in groups:
            rep_idx = group[0]
            if compare_math_answers(out, stdouts[rep_idx]):
                group.append(i)
                found = True
                break
        if not found:
            groups.append([i])
            
    largest_group = []
    for group in groups:
        if len(group) > len(largest_group):
            largest_group = group
            
    final_rewards = list(scores)
    if len(largest_group) >= 2:
        for idx in largest_group:
            comp = completions[idx]
            blocks = re.findall(r'<STEP>(.*?)</STEP>', comp, re.DOTALL)
            errors_count = 0
            accumulated_blocks = []
            for block in blocks:
                code = clean_and_dedent(block)
                accumulated_blocks.append(code)
                ast_ok = False
                try:
                    ast.parse(code)
                    ast_ok = True
                except SyntaxError:
                    pass
                exec_ok = False
                if accumulated_blocks:
                    script_lines = []
                    for k, b in enumerate(accumulated_blocks):
                        if k == len(accumulated_blocks) - 1 and len(accumulated_blocks) > 1:
                            script_lines.append("\nimport sys\nsys.stdout.write('---STEP-SEPARATOR---\\n')\nsys.stdout.flush()\n")
                        script_lines.append(b)
                        script_lines.append("\n")
                    full_code = "".join(script_lines)
                    success, _, _ = execute_code(full_code)
                    if success:
                        exec_ok = True
                if not (ast_ok and exec_ok):
                    errors_count += 1
            
            discount = (0.8 ** errors_count) * (0.9 ** (len(blocks) - 1))
            consensus_bonus = MAX_CORRECTNESS_REWARD * discount
            final_rewards[idx] += consensus_bonus
            
    return final_rewards
