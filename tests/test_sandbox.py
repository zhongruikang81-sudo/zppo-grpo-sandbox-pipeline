import unittest
import sys

# Resolve repo root from this file's location so core/ is importable.
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.sandbox import extract_code, execute_code, execute_accumulated_code

class TestSandbox(unittest.TestCase):
    def test_extract_code(self):
        text = """
        Some reasoning...
        <STEP>
        ```python
        print("Hello World")
        ```
        </STEP>
        Final thoughts...
        """
        extracted = extract_code(text)
        self.assertEqual(extracted, 'print("Hello World")')

    def test_execute_code_success(self):
        code = 'print("Answer is 42")'
        success, stdout, stderr = execute_code(code)
        self.assertTrue(success)
        self.assertEqual(stdout.strip(), "Answer is 42")
        self.assertEqual(stderr.strip(), "")

    def test_execute_code_syntax_error(self):
        code = 'print("Answer is 42' # missing quote
        success, stdout, stderr = execute_code(code)
        self.assertFalse(success)
        self.assertIn("SyntaxError", stderr)

    def test_execute_code_timeout(self):
        code = 'import time\ntime.sleep(5)'
        success, stdout, stderr = execute_code(code, timeout=1.0)
        self.assertFalse(success)
        self.assertIn("Timeout", stderr)

    def test_execute_accumulated_code_string(self):
        text_history = """
        User: Solve this problem.
        Assistant: Let's start.
        <STEP>
        a = 10
        b = 20
        </STEP>
        Observation: [none]
        Assistant: Next step.
        <STEP>
        c = a + b
        print(c)
        </STEP>
        """
        success, stdout, stderr = execute_accumulated_code(text_history)
        self.assertTrue(success)
        self.assertEqual(stdout.strip(), "30")
        self.assertEqual(stderr.strip(), "")

    def test_execute_accumulated_code_list(self):
        messages = [
            {"role": "user", "content": "Start"},
            {"role": "assistant", "content": "Let's define a.\n<STEP>\na = 5\n</STEP>"},
            {"role": "user", "content": "Observation: None"},
            {"role": "assistant", "content": "Let's multiply a by 5.\n<STEP>\nprint(a * 5)\n</STEP>"}
        ]
        success, stdout, stderr = execute_accumulated_code(messages)
        self.assertTrue(success)
        self.assertEqual(stdout.strip(), "25")
        self.assertEqual(stderr.strip(), "")

    def test_execute_accumulated_code_syntax_error_resilience(self):
        # Turn 1 has a syntax error, Turn 2 is a complete rewrite which compiles successfully
        messages = [
            {"role": "user", "content": "Start"},
            {"role": "assistant", "content": "Let's define a.\n<STEP>\na = \n</STEP>"},  # SyntaxError
            {"role": "user", "content": "Observation: SyntaxError"},
            {"role": "assistant", "content": "Correction:\n<STEP>\na = 10\nprint(a)\n</STEP>"}
        ]
        success, stdout, stderr = execute_accumulated_code(messages)
        self.assertTrue(success)
        self.assertEqual(stdout.strip(), "10")
        self.assertEqual(stderr.strip(), "")

if __name__ == "__main__":
    unittest.main()
