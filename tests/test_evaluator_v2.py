import unittest
import os
import re
import sys

# Resolve repo root from this file's location so core/ is importable.
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.evaluator import parse_ds_score, evaluate_multi_turn_completion, get_prm_salvage_score

class TestEvaluatorV2(unittest.TestCase):
    def test_parse_ds_score(self):
        # 1. Standard casing and spacing
        self.assertEqual(parse_ds_score("<score>0.30</score>"), 0.30)
        self.assertEqual(parse_ds_score("<SCORE>0.15</SCORE>"), 0.15)
        self.assertEqual(parse_ds_score("  <score>  0.40  </score>  "), 0.40)
        self.assertEqual(parse_ds_score("  <score>  0.50  </score>  "), 0.50)
        self.assertEqual(parse_ds_score("<score>0.0</score>"), 0.0)
        
        # 2. Score clamping and boundaries
        self.assertEqual(parse_ds_score("<score>0.60</score>"), 0.50) # capped at 0.50
        self.assertEqual(parse_ds_score("<score>-0.1</score>"), 0.0)  # floored at 0.0
        
        # 3. Fallbacks
        self.assertEqual(parse_ds_score("[score] 0.15"), 0.15)
        self.assertEqual(parse_ds_score("The final score is 0.30"), 0.30)
        self.assertEqual(parse_ds_score("The final score is 0.50"), 0.50)
        
        # 4. Bad values
        self.assertEqual(parse_ds_score("No score tags here"), 0.0)
        self.assertEqual(parse_ds_score("<score>abc</score>"), 0.0)

    def test_compare_math_answers_percent(self):
        from core.evaluator import compare_math_answers
        # Percent sign stripping tests
        self.assertTrue(compare_math_answers("33.33%", "33.33333333333333"))
        self.assertTrue(compare_math_answers("0.33", "0.33%"))
        self.assertTrue(compare_math_answers("  33.33%  ", "33.33"))
        self.assertTrue(compare_math_answers("33.33", "33.33%"))
        self.assertFalse(compare_math_answers("33.33", "0.3333")) # Scaling by 100 is NOT handled by rules to prevent loopholes

    def test_evaluate_multi_turn_completion_with_bypass(self):
        # Scenario 1: Model has a valid STEP block but also has raw ```python outside
        completion_with_bypass = """
        Here is the first step:
        <STEP>
        x = 100 + 23
        print(x)
        </STEP>
        And here is a bypassed code block:
        ```python
        print(456)
        ```
        """
        # It should trigger formatting penalty of -0.15
        score, stdout = evaluate_multi_turn_completion(completion_with_bypass, target_answer="123")
        
        # Calculate expected score:
        # 1 block: AST/Exec ok -> step reward = +0.1
        # Correctness matching "123" -> +0.9 * discount(errors=0, K=1) -> +0.9
        # Bypass penalty -> -0.15
        # Newline penalties: 0.0
        # Length penalty: -0.00001 * len
        expected_score_without_len = 0.1 + 0.9 - 0.15
        self.assertAlmostEqual(score, expected_score_without_len, delta=0.01)

    def test_evaluate_multi_turn_completion_without_bypass(self):
        # Scenario 2: Model has code block correctly wrapped in STEP tags
        completion_clean = """
        Here is the first step:
        <STEP>
        x = 100 + 23
        print(x)
        </STEP>
        """
        # No bypass penalty should apply
        score, stdout = evaluate_multi_turn_completion(completion_clean, target_answer="123")
        expected_score_without_len = 0.1 + 0.9
        self.assertAlmostEqual(score, expected_score_without_len, delta=0.01)

    def test_offline_salvage_fallback(self):
        # Ensure that when API key is not present (or empty), the function returns 0.0
        # and evaluate_multi_turn_completion runs without crashing
        old_env_key = os.environ.get("DEEPSEEK_API_KEY", None)
        if "DEEPSEEK_API_KEY" in os.environ:
            del os.environ["DEEPSEEK_API_KEY"]
            
        try:
            score = get_prm_salvage_score("Some completion", "42", "What is the answer?")
            self.assertEqual(score, 0.0)
            
            completion_wrong_answer = """
            <STEP>
            print(10)
            </STEP>
            """
            # Wrong answer (target 42, printed 10), should trigger salvage
            # Since API key is deleted, it should fallback to 0.0 correctness reward
            score, stdout = evaluate_multi_turn_completion(completion_wrong_answer, target_answer="42")
            expected_score_without_len = 0.1 + 0.0 # no salvage reward
            self.assertAlmostEqual(score, expected_score_without_len, delta=0.01)
        finally:
            # Restore environment key if existed
            if old_env_key is not None:
                os.environ["DEEPSEEK_API_KEY"] = old_env_key

    def test_is_suspicious_code(self):
        from core.evaluator import is_suspicious_code
        # 1. Suspicious: prints constant directly
        self.assertTrue(is_suspicious_code("print(375.0)"))
        self.assertTrue(is_suspicious_code("print('375')"))
        self.assertTrue(is_suspicious_code("x = 375.0\nprint(x)"))
        
        # 2. Not Suspicious: has math or logic
        self.assertFalse(is_suspicious_code("print(25 * 15)"))
        self.assertFalse(is_suspicious_code("x = 25 * 15\nprint(x)"))
        self.assertFalse(is_suspicious_code("import math\nprint(math.pi)"))
        self.assertFalse(is_suspicious_code("for i in range(10):\n    print(i)"))
        self.assertFalse(is_suspicious_code("def calc(): return 375.0\nprint(calc())"))

if __name__ == "__main__":
    unittest.main()
