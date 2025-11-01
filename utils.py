import json
from qrafti import run_code_in_subprocess
from datetime import datetime

dates_ = dict(start_date='2020-01-01', end_date='2024-12-31')

def log_message(payload: str, tool_name: str, code: str, mode: str = "a"):
    """Log a message to the console and a log file."""
    code = "\n".join([line for line in code.splitlines() if len(line) and 'import ' not in line])
    message = f"Output: {payload}\nTool: {tool_name}{' '*10}Date: {str(datetime.now())[:19]}\n{code}\n\n{'-'*60}\n\n"
    with open("mcp_server.log", "a") as f:
        f.write(message)
    print(message)

# log_message(f"MCP server started on {str(datetime.now())}", mode="w")

def execute_in_sandbox(code_str: str) -> str:
    """
    Safely execute a Python code string in a sandbox and return the output as JSON string.
    """
    stdout, stderr, exit_code = run_code_in_subprocess(code_str)
    print('Exit code:', exit_code)
    print(stderr)
    if exit_code:
        return json.dumps({"exit_code": exit_code, "error": stderr.strip()})
    else:
        return stdout.strip()

def _log_and_execute(tool_name: str, code: str) -> str:
    """Helper to log generated code and execute it in the sandbox."""
    payload = execute_in_sandbox(code)
    log_message(str(payload), tool_name, code)
    return payload
