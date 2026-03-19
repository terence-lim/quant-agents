# python coding_server.py
from mcp.server.fastmcp import FastMCP
import json
from server_utils import run_code_in_subprocess, log_code
import logging
#logging.basicConfig(level=logging.INFO)

#import logging
logging.basicConfig(level=logging.DEBUG)

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
    log_code(code_str)
    print(code_str)
    stdout, stderr, exit_code = run_code_in_subprocess(code_str)
    if exit_code:
        return json.dumps({"exit_code": exit_code, "error_message": stderr.strip()})
    else:
        logging.warning(stdout)
        print('STDOUT:', stdout)
        return stdout

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

