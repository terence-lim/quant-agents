import json
from datetime import datetime

def log_message(output: str = '', tool: str = '', code: str = '', mode: str = "a"):
    """Log a message to the console and a log file."""
    message = ['\n' + ("-"*60)]
    if output:
        message.append(f"Output: {output}")
    if tool:
        message.append(f"Tool: {tool}")
    if code:
        message.append(f"Code:\n{code}")
    message = "\n".join(message) + f"\nDate: {str(datetime.now())[:19]}\n" 
    with open("mcp_server.log", mode) as f:
        f.write(message)
    f.flush()
    print(message)

# log_message(f"MCP server started on {str(datetime.now())}", mode="w")

def _log_and_execute(tool_name: str, code: str) -> str:
    """Helper to log generated code and execute it in the sandbox."""
    payload = execute_in_sandbox(code)
    log_message(str(payload), tool_name, code)
    return payload
