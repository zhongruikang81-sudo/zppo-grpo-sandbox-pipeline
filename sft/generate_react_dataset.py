import os
import sys
import json
import re
from openai import OpenAI

# Resolve repo root from this file's location so core/ is importable
# regardless of the current working directory.
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.sandbox import execute_accumulated_code

# API Configurations — credentials are read from the environment; never
# hardcode API keys in source code.
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL = os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-chat")  # DeepSeek chat model

if not API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY environment variable is not set. "
        "ReAct dataset generation requires a valid DeepSeek API key."
    )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def safe_print(text: str):
    """
    Prints text safely to console, handling UnicodeEncodeError on Windows.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

def generate_sample_react(problem: str) -> list:
    """
    Interacts with DeepSeek API and local python sandbox to generate
    a multi-turn interactive ReAct dialogue trace for a given problem.
    """
    system_prompt = (
        "You are a mathematical assistant. You solve math problems step-by-step "
        "by writing Python code wrapped inside <STEP>\n```python\n# code\n```\n</STEP> blocks. "
        "After writing a code block, you MUST wait for the user to provide the observation. "
        "Once you receive the observation, you continue your reasoning. "
        "Ensure you print the intermediate variables and the final answer. "
        "When you are done, output the final answer clearly."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem}
    ]
    
    max_turns = 3
    for turn in range(max_turns):
        safe_print(f"    Turn {turn + 1}...")
        try:
            # Call DeepSeek API
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2, # low temperature for deterministic coding logic
                max_tokens=512
            )
            assistant_content = response.choices[0].message.content
            safe_print(f"      Assistant response length: {len(assistant_content)}")
            
            # Append assistant response to history
            messages.append({"role": "assistant", "content": assistant_content})
            
            # Check if assistant generated a step code block
            if "<STEP>" in assistant_content:
                # Run the sandbox on accumulated code
                success, stdout, stderr = execute_accumulated_code(messages)
                observation = stdout.strip() if success else stderr.strip()
                if not observation:
                    observation = "[Success: No output]" if success else "[Error: Unknown error]"
                
                safe_print(f"      Sandbox Observation: {observation[:100]}...")
                # Feed observation back as user message
                messages.append({"role": "user", "content": f"Observation:\n{observation}\n"})
            else:
                # Finished reasoning without code block
                safe_print("      No code block found. Reasoning finished.")
                break
        except Exception as e:
            safe_print(f"      Error calling API: {e}")
            break
            
    return messages

def main():
    # NOTE: math_pot_train.jsonl (PoT 种子数据) 与本脚本生成的两个数据集
    # 均不随仓库分发；种子数据需自备，输出写入仓库 data/ 目录。
    pot_dataset_path = str(REPO_ROOT / "data" / "math_pot_train.jsonl")
    sft_output_path = str(REPO_ROOT / "data" / "math_sft_multiturn.jsonl")
    grpo_output_path = str(REPO_ROOT / "data" / "math_grpo_dataset.jsonl")
    
    safe_print(f"Loading seed problems from local dataset {pot_dataset_path}...")
    
    if not os.path.exists(pot_dataset_path):
        safe_print(f"Error: Local dataset {pot_dataset_path} not found.")
        return
        
    seed_problems = []
    with open(pot_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            messages = data.get("messages", [])
            # Extract user message (the question)
            for msg in messages:
                if msg["role"] == "user":
                    question = msg["content"]
                    # Clean system prompt prefix if SFT system prompt was concatenated
                    if "You are a mathematical assistant" in question:
                        parts = question.split("\n\n")
                        if len(parts) > 1:
                            question = "\n\n".join(parts[1:])
                    seed_problems.append(question)
                    break
                    
    safe_print(f"Loaded {len(seed_problems)} seed problems from disk.")
    
    # Determine resume index based on existing output file
    start_idx = 0
    if os.path.exists(sft_output_path):
        try:
            with open(sft_output_path, "r", encoding="utf-8") as f_out:
                start_idx = len(f_out.readlines())
        except Exception as e:
            safe_print(f"Warning reading output file for resume: {e}")
            
    safe_print(f"Resume configuration: starting from sample index {start_idx + 1}...")
    
    num_to_generate = min(200, len(seed_problems))
    
    for idx in range(start_idx, num_to_generate):
        problem = seed_problems[idx]
        safe_print(f"\nSample {idx + 1}/{num_to_generate}:")
        safe_print(f"  Problem: {problem[:150]}...")
        
        # Generate multi-turn trace
        messages = generate_sample_react(problem)
        
        # Save SFT data
        with open(sft_output_path, "a", encoding="utf-8") as f_sft:
            f_sft.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            
        # Try to extract the final answer from the last assistant message
        final_ans = ""
        for msg in reversed(messages):
            if msg["role"] == "user" and msg["content"].startswith("Observation:"):
                final_ans = msg["content"].replace("Observation:\n", "").strip()
                break
                
        # Save GRPO prompt-answer pair
        grpo_data = {
            "prompt": problem,
            "answer": final_ans
        }
        with open(grpo_output_path, "a", encoding="utf-8") as f_grpo:
            f_grpo.write(json.dumps(grpo_data, ensure_ascii=False) + "\n")
            
        safe_print(f"  Sample {idx + 1} processed and saved successfully.")
        
    safe_print("\nDataset generation completed successfully! The datasets have been saved.")

if __name__ == "__main__":
    main()
