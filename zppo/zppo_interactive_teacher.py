import os
import sys
import re
import json
import openai
from typing import List, Dict, Tuple

# Resolve repo root from this file's location so core/ is importable
# regardless of the current working directory.
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from core.sandbox import execute_accumulated_code

# DeepSeek Configuration
# The API key is read from the DEEPSEEK_API_KEY environment variable; never
# hardcode credentials in source code.
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
# Teacher model: read from DEEPSEEK_JUDGE_MODEL, default "deepseek-chat".
# NOTE: the original hardcoded "deepseek-v4-pro" is NOT a valid DeepSeek model
# name; teacher rollouts require a valid model name.
TEACHER_MODEL = os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-chat")

def get_teacher_trajectory(question: str, target_answer: str, max_turns: int = 3) -> Tuple[bool, List[Dict[str, str]], str]:
    """
    Interacts with the DeepSeek API teacher and the local sandbox for up to max_turns.
    Returns (success, trajectory_messages, final_response)
    """
    if not API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY environment variable is not set. "
            "Teacher rollouts require a valid DeepSeek API key; "
            "set DEEPSEEK_API_KEY before running."
        )
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    prompt = (
        "You are a mathematical assistant. Use Python code wrapped inside <STEP>\n"
        "```python\n"
        "# code\n"
        "```\n"
        "</STEP> to solve the problem. Note: Never use input() in Python code. Ensure all variables are defined before use in the current step.\n\n"
        f"{question}"
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    print(f"[Teacher Interactive] Starting teacher rollout for question: {question[:80]}...")
    
    for turn in range(max_turns):
        print(f"  [Teacher Turn {turn+1}] Calling DeepSeek...")
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=messages,
                temperature=0.0, # Greedy teacher for correctness
                max_tokens=1024
            )
            assistant_content = response.choices[0].message.content
            print(f"    Teacher response length: {len(assistant_content)} characters.")
        except Exception as e:
            print(f"    [Error] Teacher API call failed: {e}")
            return False, [], str(e)
            
        messages.append({"role": "assistant", "content": assistant_content})
        
        has_code = "</STEP>" in assistant_content
        if has_code:
            # Execute in sandbox
            success, stdout, stderr = execute_accumulated_code(messages)
            observation = stdout.strip() if success else stderr.strip()
            if not observation:
                observation = "[Success: No output]" if success else "[Error: Unknown error]"
                
            print(f"    Executed code. Success: {success} | Observation length: {len(observation)}")
            messages.append({"role": "user", "content": f"Observation:\n{observation}\n"})
        else:
            print("    No code block generated. Rollout complete.")
            break
            
    # Compile final trajectory text
    final_parts = []
    for m in messages[1:]:
        if m["role"] == "assistant":
            final_parts.append(m["content"])
        else:
            final_parts.append(f"\n{m['content']}")
    final_completion = "".join(final_parts)
    
    return True, messages, final_completion

if __name__ == "__main__":
    # Quick test
    test_q = "If a box contains 3 red balls and 4 blue balls, what is the probability of picking a red ball?"
    success, traj, completion = get_teacher_trajectory(test_q, "3/7")
    print(f"Test run success: {success}")
    if success:
        print("Trajectory length:", len(traj))
        print("Final completion snippet:\n", completion[-200:])
