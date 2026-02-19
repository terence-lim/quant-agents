# python coding_server.py
from mcp.server.fastmcp import FastMCP
import json
from server_utils import log_message, run_code_in_subprocess

#import logging
#logging.basicConfig(level=logging.DEBUG)

# Create an MCP server
mcp = FastMCP("coding-server", host="0.0.0.0", port=8003)

@mcp.tool()
def execute_python(code_str: str) -> str:
    """
    Execute a Python code string and returns its standard output as a string

    Args:
        code_str (str): The Python code to execute.
    Returns:
        str: The standard output from the executed code, preferably in JSON format.
    """
    log_message(tool='execute_python', code=code_str)
    stdout, stderr, exit_code = run_code_in_subprocess(code_str)
    log_message(f"{stdout=}, {stderr=}, {exit_code=}", tool='execute_python')
    # print('Exit code:', exit_code)
    # print(stderr)
    if exit_code:
        return json.dumps({"exit_code": exit_code, "error": stderr.strip()})
    else:
        return stdout.strip()

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

