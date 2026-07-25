import json
import os
import shutil
import re
from pathlib import Path

# Repo-local data paths (this script lives in <repo>/dpo/).
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_PATH = str(_DATA_DIR / "dpo_anti_repetition.jsonl")
BACKUP_PATH = str(_DATA_DIR / "dpo_anti_repetition_original.jsonl")

print("Starting DPO Dataset Cleaning...")

# 1. Back up the original file if not already backed up
if not os.path.exists(BACKUP_PATH):
    print(f"Creating backup of original dataset to {BACKUP_PATH}")
    shutil.copyfile(INPUT_PATH, BACKUP_PATH)
else:
    print(f"Original backup already exists at {BACKUP_PATH}")

# Load original rows
with open(BACKUP_PATH, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(lines)} original pairs from backup.")

DISCARD = {
    3, 4, 6, 11, 18, 21, 22, 25, 27, 30, 32, 34, 36, 37, 38, 44, 50, 51, 52,
    57, 62, 63, 66, 70, 72, 76, 81, 82
}

KEEP = {
    1, 2, 7, 8, 9, 13, 19, 20, 24, 28, 29, 31, 35, 59, 60, 61, 64, 68, 74, 75,
    77, 79, 80, 83
}

CLEAN = {
    5, 10, 12, 14, 15, 16, 17, 23, 26, 33, 39, 40, 41, 42, 43, 45, 46, 47, 48,
    49, 53, 54, 55, 56, 58, 65, 67, 69, 71, 73, 78
}

targets = {
    5: "Juan has **498** marbles.",
    10: "- Difference: 47 - 41 = 6 cards",
    12: "**Final Answer: (B) 70**",
    14: "Hillary is left with **25 dollars** after the deposit.",
    15: "Tom made 12 dollars washing cars.",
    16: "Difference: 17,600 - 1,245 = 16,355",
    17: "- `print(result)`: prints the result.",
    23: "(7/8) / (1/16) = 14.",
    26: "4 nickels",
    33: "makes a profit of $205.",
    39: "difference in digits is indeed 3.",
    40: "John pays $650 for 3 nights at $250 per night with a $100 discount.",
    41: "399",
    42: "40 + 12 = 52",
    43: "≈ 93.3%",
    45: "The calculation shows that **4 students** are catching up on homework.",
    46: "3780",
    47: "Each gets 4 / 2 = 2 hours.",
    48: "72.60",
    49: "152",
    53: "2000",
    54: "contradiction",
    55: "134",
    56: "2 hours",
    58: "Observation:\n4",
    65: "Let me know if you have other problems you'd like me to solve.",
    67: "So the answer is **81 pounds**.",
    69: "Therefore, the correct answer is **(A) 6.4**.",
    71: "Let me know if you have any other calculations you'd like to perform!",
    73: "Last month books: 4\nThis month books: 8\nTotal books for two months: 12",
    78: "equal to **0.8**.",
}

def get_truncation_index(idx, text):
    target = targets.get(idx)
    if not target:
        return -1
    pos = text.find(target)
    if pos != -1:
        return pos + len(target)
    return -1

cleaned_dataset = []
discarded_count = 0
kept_count = 0
cleaned_count = 0

for idx, item in enumerate(lines, 1):
    if idx in DISCARD:
        discarded_count += 1
        continue
    
    # Create a copy of the item to modify
    new_item = {
        "prompt": item["prompt"],
        "chosen": item["chosen"],
        "rejected": item["rejected"],
        "source": item.get("source", ""),
        "step": item.get("step", 0)
    }
    
    if idx in CLEAN:
        pos = get_truncation_index(idx, item["chosen"])
        if pos != -1:
            new_item["chosen"] = item["chosen"][:pos].strip()
            cleaned_count += 1
        else:
            print(f"Warning: Truncation point for Row {idx} not found during final processing! Keeping original chosen.")
            new_item["chosen"] = item["chosen"].strip()
            cleaned_count += 1
    elif idx in KEEP:
        new_item["chosen"] = item["chosen"].strip()
        kept_count += 1
        
    new_item["rejected"] = item["rejected"].strip()
    cleaned_dataset.append(new_item)

# 2. Save the cleaned dataset back to data/dpo_anti_repetition.jsonl
with open(INPUT_PATH, "w", encoding="utf-8") as f:
    for item in cleaned_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\n--- Cleaning Complete ---")
print(f"Total processed: {len(lines)}")
print(f"  Discarded: {discarded_count}")
print(f"  Kept as is: {kept_count}")
print(f"  Cleaned & Used: {cleaned_count}")
print(f"  Final Cleaned Dataset size: {len(cleaned_dataset)} pairs")
print(f"Saved cleaned dataset to: {INPUT_PATH}")
