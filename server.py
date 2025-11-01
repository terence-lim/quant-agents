"""Shared MCP server exposing metadata utilities for research agents."""
from mcp.server.fastmcp import FastMCP
from qrafti import load_variables
from qrafti import run_code_in_subprocess
import json

dates = dict(start_date='2020-01-01', end_date='2024-12-31')

def log_message(message: str, mode: str = "a"):
    """Log a message to the console and a log file."""
    with open("mcp_server.log", "a") as f:
        f.write(message + "\n")
    print(message)
# log_message(f"MCP server started on {str(datetime.now())}", mode="w")

# Create an MCP server

mcp = FastMCP("metadata-server", host="0.0.0.0", port=8002)

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

@mcp.tool()
def Panel_matmul(panel_id: str, other_panel_id: str) -> str:
    """Compute the dot product (matrix multiplication) between two Panels.
    Args:
        panel_id (str): The id of the first panel data set.
        other_panel_id (str): The id of the second panel data set.
    Returns:
        str: the id of the created Panel in the cache in JSON format
    """
    code = f"""
import json
from qrafti import Panel
p1, p2 = Panel('{panel_id}', **{dates}), Panel('{other_panel_id}', **{dates})
p3 = (p1 @ p2).persist()
print(json.dumps({{'result_panel_id': p3.name, 'metadata': p3.info}}))
"""
    log_message(f"\\nExecuting code for Panel_matmul:\\n{code}\\n")
    return execute_in_sandbox(code)


@mcp.tool()
def get_variables_descriptions() -> dict:
    """Return a mapping of Panel identifiers to their descriptions."""
    df = load_variables()
    if "Description" not in df.columns:
        return {}
    return df["Description"].to_dict()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
