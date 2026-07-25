import json
import os
from pathlib import Path

# Repo-local paths: this script lives in <repo>/data/
DATA_DIR = Path(__file__).resolve().parent

def main():
    dataset_path = str(DATA_DIR / "numina_gsm_mix_numeric.jsonl")
    output_path = str(DATA_DIR / "numina_500_test.json")
    
    print(f"Reading dataset from {dataset_path}...")
    test_questions = []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                # Extract index range 1980 to 2479 (exactly 500 questions)
                if 1980 <= idx < 2480:
                    data = json.loads(line)
                    test_questions.append({
                        "index": idx,
                        "prompt": data["prompt"],
                        "answer": data["answer"]
                    })
                    
    print(f"Extracted {len(test_questions)} questions (indices {test_questions[0]['index']} to {test_questions[-1]['index']}).")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_questions, f, indent=2, ensure_ascii=False)
        
    print(f"Saved test dataset to {output_path}")

if __name__ == "__main__":
    main()
