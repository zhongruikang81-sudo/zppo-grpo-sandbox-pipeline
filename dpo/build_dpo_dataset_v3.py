import os
import re
import json
import sys
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding='utf-8')

# Resolve repo root from this file's location (this script lives in <repo>/dpo/).
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent

# DeepSeek credentials are read from the environment; never hardcode API keys.
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
# NOTE: the original pipeline also used an invalid judge model name
# ("deepseek-v4-pro"); a valid model name is required. Override via
# DEEPSEEK_JUDGE_MODEL if needed.
JUDGE_MODEL = os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-chat")

def correct_with_deepseek(question, target_answer, buggy_body):
    if not API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY environment variable is not set. "
            "DeepSeek-based code correction requires a valid API key."
        )
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=45.0)
    
    system_prompt = """You are a Python programming and mathematics expert.
Your task is to take a mathematical question, its correct target answer, and a buggy multi-turn student solution that has a coding error, and output the CORRECTED student solution.

The corrected solution must:
1. Keep the exact same writing style, explanations, and structure as the original student solution.
2. Fix the Python code inside the <STEP>...</STEP> blocks. Common bugs to fix include:
   - Replacing mathematical notation like 'choose' with 'math.comb' or 'sympy.comb'.
   - Replacing '^' (bitwise XOR) with '**' (exponentiation).
   - Fixing NameError by defining variables.
   - Fixing SyntaxError like 'return' outside functions.
3. Update the 'Observation:' block to show the correct standard output of the corrected code.
4. Update the subsequent assistant reasoning and final answer in the text to reflect the correct output.

Output ONLY the corrected student solution, starting from the first word of the solution. Do not include any meta-talk or wrappers like ```markdown or ```text."""

    user_content = f"""【Mathematical Question】
{question}

【Target Answer】
{target_answer}

【Buggy Student Solution】
{buggy_body}

Corrected student solution:"""

    import time
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=1500,
                timeout=50.0
            )
            res = response.choices[0].message.content
            if res and len(res.strip()) > 0:
                return res.strip()
            raise ValueError("Empty response content from API")
        except Exception as e:
            print(f"[API Error/Attempt {attempt+1}] Failed to correct buggy code for question '{question[:30]}...': {e}")
            if attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 3
                time.sleep(sleep_time)
            else:
                return None

def main():
    log_path = str(REPO_ROOT / "grpo_output_discounted" / "training_log_v2.txt")
    out_path = str(REPO_ROOT / "data" / "dpo_formatting_alignment.jsonl")
    
    if not os.path.exists(log_path):
        print("Log file not found.")
        return
        
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    step_blocks = []
    matches = list(re.finditer(r'--- Step (\d+) ---', text))
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        step_blocks.append(text[start:end])
        
    print(f"Total steps in log: {len(step_blocks)}")
    
    # Filter Level 3 steps (961+)
    target_steps = [block for block in step_blocks if re.search(r'--- Step (\d+) ---', block) and int(re.search(r'--- Step (\d+) ---', block).group(1)) >= 961]
    
    # Group candidates globally by normalized question
    question_to_cands = {}
    for block in target_steps:
        q_match = re.search(r'Question:\s*(.*?)(?=\nTarget Answer:)', block, re.DOTALL)
        if not q_match:
            continue
        q = q_match.group(1).strip()
        norm_q = "".join(q.split())
        
        ta_match = re.search(r'Target Answer:\s*(.*?)(?=\n)', block)
        target = ta_match.group(1).strip() if ta_match else ""
        
        cands = re.split(r'--- Candidate \d+ \(Reward: .*?\) ---', block)
        cand_headers = re.findall(r'--- Candidate (\d+) \(Reward: (.*?)\) ---', block)
        
        if not cand_headers or len(cands) - 1 != len(cand_headers):
            continue
            
        if norm_q not in question_to_cands:
            question_to_cands[norm_q] = {
                "prompt": q,
                "target": target,
                "correct_code": [],
                "nocode_wrong": [],
                "buggy_code": []
            }
            
        group = question_to_cands[norm_q]
        for idx, (c_num, c_rew) in enumerate(cand_headers):
            rew_val = float(c_rew)
            body = cands[idx + 1].strip()
            has_code = "<STEP>" in body
            
            obs_blocks = body.split("Observation:")
            last_obs = obs_blocks[-1].strip() if len(obs_blocks) > 1 else ""
            is_error = any(err in last_obs for err in ["Error", "Exception", "Traceback", "invalid syntax"])
            
            if rew_val >= 0.80 and has_code:
                group["correct_code"].append((rew_val, body))
            elif not has_code and rew_val <= 0.0:
                group["nocode_wrong"].append(body)
            elif has_code and (rew_val <= 0.0 or is_error):
                group["buggy_code"].append(body)
                
    dpo_pairs = []
    buggy_to_correct = [] # items to correct via DeepSeek: (question, target, buggy_body)
    
    type1_count = 0
    
    for norm_q, group in question_to_cands.items():
        if group["correct_code"]:
            # Pick the best correct code solution as chosen
            group["correct_code"].sort(key=lambda x: x[0], reverse=True)
            chosen_body = group["correct_code"][0][1]
            
            # Type 1: Wrong Mental Math vs Correct Code
            for nocode_body in group["nocode_wrong"]:
                dpo_pairs.append({
                    "prompt": group["prompt"],
                    "chosen": chosen_body,
                    "rejected": nocode_body,
                    "type": "force_code"
                })
                type1_count += 1
                
            # Type 2: Collect buggy code runs to correct via DeepSeek
            for buggy_body in group["buggy_code"]:
                buggy_to_correct.append((group["prompt"], group["target"], buggy_body))

    print(f"Mined {type1_count} Type 1 (Wrong Mental Math vs Correct Code) pairs.")
    print(f"Mined {len(buggy_to_correct)} buggy code candidates to correct via DeepSeek API.")
    
    if buggy_to_correct:
        print("\nStarting parallel DeepSeek code correction for Type 2 DPO pairs...")
        # Use ThreadPoolExecutor for fast parallel API calls
        type2_count = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_bug = {
                executor.submit(correct_with_deepseek, q, target, buggy): (q, target, buggy)
                for q, target, buggy in buggy_to_correct
            }
            
            for future in as_completed(future_to_bug):
                q, target, buggy = future_to_bug[future]
                corrected_body = future.result()
                if corrected_body:
                    dpo_pairs.append({
                        "prompt": q,
                        "chosen": corrected_body,
                        "rejected": buggy,
                        "type": "correct_buggy_code"
                    })
                    type2_count += 1
                    if type2_count % 5 == 0:
                        print(f"  Corrected {type2_count}/{len(buggy_to_correct)} buggy candidates...")
                        
        print(f"Successfully generated {type2_count} Type 2 (Corrected Code vs Buggy Code) pairs.")

    # Save final DPO dataset
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
    print(f"\nDPO alignment dataset successfully saved to: {out_path}")
    print(f"Total dataset size: {len(dpo_pairs)} pairs (Type 1: {type1_count}, Type 2: {len(dpo_pairs) - type1_count})")

if __name__ == "__main__":
    main()
