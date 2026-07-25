import os
import sys
import json
import random
import re
from typing import List, Dict, Any, Tuple

# Add this script's directory (for zppo_interactive_teacher) and the repo root
# (for core/ imports) to sys.path.
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from zppo_interactive_teacher import get_teacher_trajectory

def extract_core_question(prompt: str) -> str:
    """
    Strips away standard SFT system prefix to get the raw question text.
    """
    if "You are a mathematical assistant" in prompt:
        parts = prompt.split("\n\n")
        if len(parts) > 1:
            return "\n\n".join(parts[1:])
    return prompt

def clean_trajectory(traj: str) -> str:
    """
    Extracts only the code blocks and observations to save tokens and prevent OOM.
    """
    cleaned_parts = []
    step_matches = list(re.finditer(r'(<STEP>.*?</STEP>)', traj, re.DOTALL))
    
    for i, match in enumerate(step_matches):
        step_code = match.group(1)
        cleaned_parts.append(step_code)
        
        start_idx = match.end()
        end_idx = step_matches[i+1].start() if i + 1 < len(step_matches) else len(traj)
        search_space = traj[start_idx:end_idx]
        
        obs_match = re.search(r'(Observation:\s*.*?\n)(?=\n|<STEP>|$)', search_space, re.DOTALL)
        if obs_match:
            cleaned_parts.append(obs_match.group(1).strip())
        else:
            obs_match = re.search(r'(Observation:\s*[^\n]+)', search_space)
            if obs_match:
                cleaned_parts.append(obs_match.group(1).strip())
                
    if cleaned_parts:
        return "\n".join(cleaned_parts)
    return traj

class PromptReplayBuffer:
    def __init__(self, dataset_path: str, buffer_size: int = 16, grad_threshold: int = 1, max_steps: int = 5, start_index: int = 0):
        self.dataset_path = dataset_path
        self.buffer_size = buffer_size
        self.grad_threshold = grad_threshold
        self.max_steps = max_steps
        
        self.dataset = []
        self.load_dataset()
        
        self.pool_ptr = start_index
        self.buffer: List[Dict[str, Any]] = []
        self.current_step = 0
        
    def load_dataset(self):
        print(f"[PRB] Loading dataset from {self.dataset_path}...")
        if self.dataset_path.endswith(".jsonl"):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.dataset.append(json.loads(line))
        elif self.dataset_path.endswith(".json"):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
        print(f"[PRB] Loaded {len(self.dataset)} questions.")
        
    def replenish(self, student_eval_fn=None):
        """
        Fills the buffer up to buffer_size by drawing new questions, running student attempts,
        and generating teacher trajectories for failed questions.
        """
        replenished = 0
        while len(self.buffer) < self.buffer_size and self.pool_ptr < len(self.dataset):
            q_data = self.dataset[self.pool_ptr]
            core_q = extract_core_question(q_data["prompt"])
            target = str(q_data["answer"])
            
            # Step 1: Run student attempt first (3-turn interaction)
            if student_eval_fn is not None:
                print(f"\n[PRB Replenish] Evaluating index {self.pool_ptr} on student model...")
                student_correct, student_traj = student_eval_fn(core_q, target)
            else:
                student_correct = False
                student_traj = "I will write a program in Python:\n<STEP>\nprint(0)\n</STEP>\nObservation:\n0\n\nTherefore, the answer is 0."
                
            if student_correct:
                print(f"  [PRB Replenish] Student solved index {self.pool_ptr} correctly on first attempt! Bypassing.")
                self.pool_ptr += 1
                continue
                
            # Step 2: Student failed. Pre-generate teacher interactive trajectory (deterministic)
            success, messages, teacher_completion = get_teacher_trajectory(core_q, target)
            
            if success:
                self.buffer.append({
                    "question_data": q_data,
                    "core_question": core_q,
                    "target": target,
                    "index": self.pool_ptr,
                    "student_errors": [student_traj],  # Seed with the actual student error!
                    "teacher_trajectory": teacher_completion,
                    "consecutive_correct": 0,
                    "steps_in_buffer": 0
                })
                replenished += 1
                print(f"  [PRB Replenish] Question index {self.pool_ptr} added to PRB with teacher trajectory and actual student error.")
            else:
                print(f"  [PRB Replenish] [Warning] Failed to generate teacher trajectory for index {self.pool_ptr}. Skipping.")
                
            self.pool_ptr += 1
            
        print(f"[PRB] Buffer replenished. Current size: {len(self.buffer)}/{self.buffer_size}")
        
    def update_buffer_item(self, index: int, student_traj: str, is_correct: bool):
        """
        Updates consecutive correct count and logs student failures for a question index.
        If student_traj is None or empty, we skip logging it as an error to keep high-quality errors.
        """
        for item in self.buffer:
            if item["index"] == index:
                item["steps_in_buffer"] += 1
                if is_correct:
                    item["consecutive_correct"] += 1
                    print(f"  [PRB Update] Q{index} Correct! Consecutive correct: {item['consecutive_correct']}/{self.grad_threshold}")
                else:
                    item["consecutive_correct"] = 0
                    if student_traj:
                        # Save the latest error trajectory
                        item["student_errors"].append(student_traj)
                        if len(item["student_errors"]) > 3:
                            item["student_errors"].pop(0)
                        print(f"  [PRB Update] Q{index} Incorrect! Error logged. consecutive_correct reset to 0.")
                    else:
                        print(f"  [PRB Update] Q{index} Incorrect (Format/No Code)! Skipped error logging to preserve high-quality errors. consecutive_correct reset to 0.")
                break
                
    def clean_and_replenish(self, student_eval_fn=None) -> Tuple[int, int]:
        """
        Removes graduated and retired questions and replenishes the buffer.
        Returns (graduated_count, retired_count)
        """
        graduated_count = 0
        retired_count = 0
        new_buffer = []
        
        for item in self.buffer:
            if item["consecutive_correct"] >= self.grad_threshold:
                print(f"  [PRB Graduation] Q{item['index']} graduated from buffer (correct threshold reached)!")
                graduated_count += 1
            elif item["steps_in_buffer"] >= self.max_steps:
                print(f"  [PRB Retirement] Q{item['index']} retired from buffer (max steps reached: {item['steps_in_buffer']})!")
                retired_count += 1
            else:
                new_buffer.append(item)
                
        self.buffer = new_buffer
        self.replenish(student_eval_fn=student_eval_fn)
        return graduated_count, retired_count
        
    def get_bcq_prompt(self, item: Dict[str, Any]) -> str:
        """
        Constructs the Bi-candidate Comparison Query (BCQ) prompt.
        """
        core_q = item["core_question"]
        y_teacher = clean_trajectory(item["teacher_trajectory"])
        
        if item["student_errors"]:
            y_student = clean_trajectory(item["student_errors"][-1])
        else:
            # Fallback if no error has been recorded yet (first run)
            # We construct a simple placeholder error to start the BCQ loop
            y_student = "I will write a program in Python:\n<STEP>\nprint(0)\n</STEP>\nObservation:\n0\n\nTherefore, the answer is 0."
            
        candidates = [
            ("Candidate A", y_student, "incorrect"),
            ("Candidate B", y_teacher, "correct")
        ]
        # Shuffle order to prevent position bias
        random.shuffle(candidates)
        
        c1_name, c1_text, _ = candidates[0]
        c2_name, c2_text, _ = candidates[1]
        
        # BCQ Prompt Template (English instructions for better compatibility with Gemma-2)
        bcq_prompt = (
            "You are a mathematical assistant. Use Python code wrapped inside <STEP>\n"
            "```python\n"
            "# code\n"
            "```\n"
            "</STEP> to solve the problem. Note: Never use input() in Python code. Ensure all variables are defined before use in the current step.\n\n"
            f"For the mathematical problem:\n{core_q}\n\n"
            f"Here are two candidate solutions:\n\n"
            f"【{c1_name}】:\n{c1_text}\n\n"
            f"【{c2_name}】:\n{c2_text}\n\n"
            f"Compare and analyze the reasoning of both candidates. Identify which candidate is incorrect and explain exactly where it went wrong. Then, write your correct step-by-step reasoning, and write Python code inside <STEP> to compute the correct final answer."
        )
        return bcq_prompt
        
    def save_state(self, path: str):
        state = {
            "pool_ptr": self.pool_ptr,
            "current_step": self.current_step,
            "buffer": self.buffer
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(f"[PRB] State saved to {path}")
        
    def load_state(self, path: str):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.pool_ptr = state["pool_ptr"]
                self.buffer = state["buffer"]
                self.current_step = state.get("current_step", 0)
                print(f"[PRB] Loaded state from {path}. pool_ptr: {self.pool_ptr}, current_step: {self.current_step}, buffer size: {len(self.buffer)}")
            except Exception as e:
                print(f"[PRB] Failed to load state: {e}. Starting fresh.")
                
if __name__ == "__main__":
    # Test buffer replenishment and BCQ construction
    db_path = str(REPO_ROOT / "data" / "numina_gsm_mix_numeric.jsonl")
    prb = PromptReplayBuffer(db_path, buffer_size=2, start_index=0)
    prb.replenish()
    if prb.buffer:
        item = prb.buffer[0]
        prompt = prb.get_bcq_prompt(item)
        print("\nGenerated BCQ Prompt snippet:\n", prompt[:300])
