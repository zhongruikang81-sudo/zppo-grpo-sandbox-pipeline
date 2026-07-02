import sys
import subprocess
import os
import re
import tempfile
import textwrap
from typing import Tuple, List, Union, Dict

def clean_and_dedent(code_block: str) -> str:
    """
    Cleans markdown code fences and correctly dedents code lines using textwrap.dedent.
    Supports cases where there is trailing text after the closing code fence.
    """
    # 1. Check for standard ```python ... ``` block
    match_fence = re.search(r'```python\s*(.*?)\s*```', code_block, re.DOTALL)
    if match_fence:
        code_block = match_fence.group(1)
    else:
        # 2. Check for general ``` ... ``` block
        match_fence_general = re.search(r'```\s*(.*?)\s*```', code_block, re.DOTALL)
        if match_fence_general:
            code_block = match_fence_general.group(1)
        else:
            # 3. Fallback: original prefix/suffix matching
            match_fence_start = re.match(r'^\s*(```python|```)', code_block)
            if match_fence_start:
                code_block = code_block[match_fence_start.end():]
                code_block = re.sub(r'```\s*$', '', code_block)
    return textwrap.dedent(code_block).strip()

def extract_code(text: str) -> str:
    """
    Extracts Python code from the last <STEP>...</STEP> block in the text.
    Strips away markdown code fences if present.
    """
    pattern = re.compile(r'<STEP>(.*?)</STEP>', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return ""
    return clean_and_dedent(matches[-1])

def execute_code(code: str, timeout: float = 2.0) -> Tuple[bool, str, str]:
    """
    Executes the Python code in a secure sub-process and returns (success, stdout, stderr).
    """
    if not code.strip():
        return False, "", "No code found to execute."
        
    # Use a local scratch directory for temp files
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch")
    os.makedirs(temp_dir, exist_ok=True)
    
    fd, temp_path = tempfile.mkstemp(suffix=".py", dir=temp_dir)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            # Prepend mock to prevent input() from hanging the sandbox subprocess
            mock_header = "import builtins\nbuiltins.input = lambda *args, **kwargs: '1'\n"
            f.write(mock_header + code)
            
        # Force UTF-8 encoding for subprocess stdout/stderr to prevent
        # UnicodeEncodeError on Windows (default GBK cannot encode π, ₂, ° etc.)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            env=env
        )
        success = (result.returncode == 0)
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", f"Execution Timeout (>{timeout}s)"
    except Exception as e:
        return False, "", str(e)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

def execute_accumulated_code(text_history: Union[str, List[Dict[str, str]]], timeout: float = 2.0) -> Tuple[bool, str, str]:
    """
    Extracts all <STEP>...</STEP> blocks from conversation history,
    concatenates them into a single Python script, executes it, and returns
    the output (stdout/stderr) of the latest block execution.
    """
    blocks = []
    if isinstance(text_history, list):
        for msg in text_history:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                found = re.findall(r'<STEP>(.*?)</STEP>', content, re.DOTALL)
                blocks.extend(found)
    else:
        blocks = re.findall(r'<STEP>(.*?)</STEP>', text_history, re.DOTALL)
        
    cleaned_blocks = []
    for block in blocks:
        code = clean_and_dedent(block)
        if code:
            cleaned_blocks.append(code)
            
    if not cleaned_blocks:
        return False, "", "No code blocks found in history."
        
    # Concatenate blocks, injecting separator before the last block
    script_lines = []
    # Filter historical blocks to exclude those with compile-time SyntaxErrors
    valid_historical_blocks = []
    for block in cleaned_blocks[:-1]:
        try:
            compile(block, '<string>', 'exec')
            valid_historical_blocks.append(block)
        except SyntaxError:
            # Skip syntactically invalid historical blocks so they don't break the entire script compilation
            pass
            
    for block in valid_historical_blocks:
        indented_block = textwrap.indent(block, "    ")
        wrapped = f"try:\n{indented_block}\n    pass\nexcept Exception:\n    pass"
        script_lines.append(wrapped)
        script_lines.append("\n")
        
    # Always append the latest block at the end
    if len(cleaned_blocks) > 1:
        script_lines.append("\nimport sys\nsys.stdout.write('---STEP-SEPARATOR---\\n')\nsys.stdout.flush()\n")
    script_lines.append(cleaned_blocks[-1])
    script_lines.append("\n")
        
    full_code = "".join(script_lines)
    success, stdout, stderr = execute_code(full_code, timeout=timeout)
    
    if "---STEP-SEPARATOR---" in stdout:
        parts = stdout.split("---STEP-SEPARATOR---")
        latest_stdout = parts[-1].lstrip("\n")
    else:
        latest_stdout = stdout
        
    return success, latest_stdout, stderr
