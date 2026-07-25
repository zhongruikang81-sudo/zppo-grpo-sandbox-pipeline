import os
import sys
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Resolve repo root from this file's location so core/ is importable.
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.sandbox import execute_accumulated_code

def test_inference_multiturn(question: str, model, tokenizer, max_turns: int = 4):
    print(f"\n========================================")
    print(f"Question: {question}")
    print(f"========================================\n")
    
    system_prompt = (
        "You are a mathematical assistant. You solve math problems step-by-step by writing Python code wrapped inside <STEP>\n"
        "```python\n"
        "# code\n"
        "```\n"
        "</STEP> blocks. After writing a code block, you MUST wait for the user to provide the observation. "
        "Once you receive the observation, you continue your reasoning. Always print the final answer at the very end."
    )
    
    # Initialize history
    history = [{"role": "user", "content": f"{system_prompt}\n\n{question}"}]
    
    turn = 1
    max_safety_turns = 15
    while turn <= max_safety_turns:
        print(f"\n--- Turn {turn} (Model Generating...) ---")
        prompt_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_str, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
            )
            
        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        assistant_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        print(f"\n[Turn {turn} Assistant Output]:")
        print(assistant_text)
        
        # Append assistant turn to history
        history.append({"role": "assistant", "content": assistant_text})
        
        # Check if the assistant wants to execute a code block
        if "</STEP>" in assistant_text or "<STEP>" in assistant_text:
            print(f"\nExecuting code blocks accumulated in history...")
            success, stdout, stderr = execute_accumulated_code(history)
            observation = stdout.strip() if success else stderr.strip()
            if not observation:
                observation = "[Success: No output]" if success else "[Error: Unknown error]"
                
            print(f"[Sandbox Output]:\n{observation}")
            
            # Append Observation to history
            history.append({"role": "user", "content": f"Observation:\n{observation}\n"})
            turn += 1
        else:
            print("\n[Stop Condition]: Assistant did not output a <STEP> code block. Ending interactive loop.")
            break
    else:
        print(f"\n[Warning]: Reached maximum safety limit of {max_safety_turns} turns. Terminating loop.")
            
    print("\n========================================")
    print("Full Conversation Transcript:")
    print("========================================")
    for msg in history:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n--- {role} ---\n{content}")
    print("========================================\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run multi-turn ReAct agent evaluation.")
    parser.add_argument("--adapter", type=str, default=str(REPO_ROOT / "grpo_output_discounted"),
                        help="Path to the PEFT adapter to load.")
    args = parser.parse_args()
    
    # NOTE: 基座模型不随仓库分发；通过 BASE_MODEL_ID 环境变量指向本地副本。
    model_id = os.environ.get("BASE_MODEL_ID", "google/gemma-2-2b-it")
    adapter_id = args.adapter
    
    print(f"Loading tokenizer from {adapter_id}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
    
    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading Peft adapter from {adapter_id}...")
    model = PeftModel.from_pretrained(base_model, adapter_id)
    model.eval()
    
    test_questions = [
        "Find the sum of all positive integers n < 1000 such that the sum of the proper divisors of n (excluding n itself) is a perfect square. Let's write a Python program to solve this.",
        "A game is played on a 2D grid where a token starts at (0,0). In each step, the token can move either to (x+1, y) or (x, y+1) with equal probability. What is the exact probability that the token reaches (4,4) without ever touching the diagonal points (1,1), (2,2), or (3,3) during the path? Let's write a Python program to compute this.",
        "Calculate the area of the region bounded by y = x^2, y = 12 - 2*x^2, and x >= 0 using Python numerical integration or symbolic math."
    ]
    
    for question in test_questions:
        test_inference_multiturn(question, model, tokenizer, max_turns=4)

if __name__ == "__main__":
    main()
