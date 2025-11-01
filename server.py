"""Shared MCP server exposing metadata utilities for research agents."""
from mcp.server.fastmcp import FastMCP
from qrafti import load_variables
from qrafti import run_code_in_subprocess
import json
from typing import Any, List, Optional


def _panel_filter_code(panel_id: str, params: dict) -> str:
    """Return sandbox code for invoking Panel.filter/fill with provided parameters."""
    serialized = json.dumps(params)
    encoded = json.dumps(serialized)
    return f"""
import json
from qrafti import Panel

args = json.loads({encoded})
panel = Panel('{panel_id}', **{dates})
mask_panel = Panel(args['mask_panel_id'], **{dates}) if args.get('mask_panel_id') else None
index_panel = Panel(args['index_panel_id'], **{dates}) if args.get('index_panel_id') else None

filtered = getattr(panel, 'fill', panel.filter)(
    start_date=args.get('start_date'),
    end_date=args.get('end_date'),
    dates=args.get('dates'),
    stocks=args.get('stocks'),
    min_stocks=args.get('min_stocks'),
    min_value=args.get('min_value'),
    max_value=args.get('max_value'),
    values=args.get('values'),
    dropna=args.get('dropna', False),
    mask=mask_panel,
    index=index_panel,
)
filtered = filtered.persist()
print(json.dumps({{'result_panel_id': filtered.name, 'metadata': filtered.info}}))
"""


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


def _log_and_execute(tool_name: str, code: str) -> str:
    """Helper to log generated code and execute it in the sandbox."""
    log_message(f"\nExecuting code for {tool_name}:\n{code}\n")
    return execute_in_sandbox(code)


def _binary_panel_operation_code(op: str, panel_id: str, other_panel_id: str) -> str:
    """Return the sandbox code for a binary Panel operation."""
    return f"""
import json
from qrafti import Panel
p1, p2 = Panel('{panel_id}', **{dates}), Panel('{other_panel_id}', **{dates})
p3 = ({op}).persist()
print(json.dumps({{'result_panel_id': p3.name, 'metadata': p3.info}}))
"""


def _unary_panel_operation_code(op: str, panel_id: str) -> str:
    """Return the sandbox code for a unary Panel operation."""
    return f"""
import json
from qrafti import Panel
p1 = Panel('{panel_id}', **{dates})
p2 = ({op}).persist()
print(json.dumps({{'result_panel_id': p2.name, 'metadata': p2.info}}))
"""


@mcp.tool()
def Panel_matmul(panel_id: str, other_panel_id: str) -> str:
    """Compute the dot product (matrix multiplication) between two Panels.
    Args:
        panel_id (str): The id of the first panel data set.
        other_panel_id (str): The id of the second panel data set.
    Returns:
        str: the id of the created Panel in the cache in JSON format
    """
    code = _binary_panel_operation_code("p1 @ p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_matmul", code)


@mcp.tool()
def Panel_add(panel_id: str, other_panel_id: str) -> str:
    """Add two Panels element-wise."""
    code = _binary_panel_operation_code("p1 + p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_add", code)


@mcp.tool()
def Panel_radd(panel_id: str, other_panel_id: str) -> str:
    """Add two Panels element-wise (reverse order)."""
    code = _binary_panel_operation_code("p1.__radd__(p2)", panel_id, other_panel_id)
    return _log_and_execute("Panel_radd", code)


@mcp.tool()
def Panel_sub(panel_id: str, other_panel_id: str) -> str:
    """Subtract the second Panel from the first Panel."""
    code = _binary_panel_operation_code("p1 - p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_sub", code)


@mcp.tool()
def Panel_rsub(panel_id: str, other_panel_id: str) -> str:
    """Subtract the first Panel from the second Panel."""
    code = _binary_panel_operation_code("p1.__rsub__(p2)", panel_id, other_panel_id)
    return _log_and_execute("Panel_rsub", code)


@mcp.tool()
def Panel_mul(panel_id: str, other_panel_id: str) -> str:
    """Multiply two Panels element-wise."""
    code = _binary_panel_operation_code("p1 * p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_mul", code)


@mcp.tool()
def Panel_rmul(panel_id: str, other_panel_id: str) -> str:
    """Multiply two Panels element-wise (reverse order)."""
    code = _binary_panel_operation_code("p1.__rmul__(p2)", panel_id, other_panel_id)
    return _log_and_execute("Panel_rmul", code)


@mcp.tool()
def Panel_truediv(panel_id: str, other_panel_id: str) -> str:
    """Divide the first Panel by the second Panel."""
    code = _binary_panel_operation_code("p1 / p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_truediv", code)


@mcp.tool()
def Panel_rtruediv(panel_id: str, other_panel_id: str) -> str:
    """Divide the second Panel by the first Panel."""
    code = _binary_panel_operation_code("p1.__rtruediv__(p2)", panel_id, other_panel_id)
    return _log_and_execute("Panel_rtruediv", code)


@mcp.tool()
def Panel_eq(panel_id: str, other_panel_id: str) -> str:
    """Compare two Panels for element-wise equality."""
    code = _binary_panel_operation_code("p1 == p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_eq", code)


@mcp.tool()
def Panel_ne(panel_id: str, other_panel_id: str) -> str:
    """Compare two Panels for element-wise inequality."""
    code = _binary_panel_operation_code("p1 != p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_ne", code)


@mcp.tool()
def Panel_lt(panel_id: str, other_panel_id: str) -> str:
    """Compare whether the first Panel is less than the second Panel element-wise."""
    code = _binary_panel_operation_code("p1 < p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_lt", code)


@mcp.tool()
def Panel_le(panel_id: str, other_panel_id: str) -> str:
    """Compare whether the first Panel is less than or equal to the second Panel element-wise."""
    code = _binary_panel_operation_code("p1 <= p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_le", code)


@mcp.tool()
def Panel_gt(panel_id: str, other_panel_id: str) -> str:
    """Compare whether the first Panel is greater than the second Panel element-wise."""
    code = _binary_panel_operation_code("p1 > p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_gt", code)


@mcp.tool()
def Panel_ge(panel_id: str, other_panel_id: str) -> str:
    """Compare whether the first Panel is greater than or equal to the second Panel element-wise."""
    code = _binary_panel_operation_code("p1 >= p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_ge", code)


@mcp.tool()
def Panel_or(panel_id: str, other_panel_id: str) -> str:
    """Combine two Panels using a logical OR operation."""
    code = _binary_panel_operation_code("p1 | p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_or", code)


@mcp.tool()
def Panel_and(panel_id: str, other_panel_id: str) -> str:
    """Combine two Panels using a logical AND operation."""
    code = _binary_panel_operation_code("p1 & p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_and", code)


@mcp.tool()
def Panel_neg(panel_id: str) -> str:
    """Negate the values of a Panel."""
    code = _unary_panel_operation_code("-p1", panel_id)
    return _log_and_execute("Panel_neg", code)


@mcp.tool()
def Panel_invert(panel_id: str) -> str:
    """Apply a logical NOT operation to a Panel."""
    code = _unary_panel_operation_code("~p1", panel_id)
    return _log_and_execute("Panel_invert", code)


@mcp.tool()
def Panel_filter(
    panel_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    dates: Optional[List[str]] = None,
    stocks: Optional[List[str]] = None,
    min_stocks: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    values: Optional[List[Any]] = None,
    mask_panel_id: Optional[str] = None,
    index_panel_id: Optional[str] = None,
    dropna: bool = False,
) -> str:
    """Filter a Panel by invoking ``Panel.fill`` (or ``Panel.filter`` when unavailable).

    Args:
        panel_id: Identifier of the Panel to filter.
        start_date: Keep rows with dates on or after this value.
        end_date: Keep rows with dates on or before this value.
        dates: Explicit list of dates to retain.
        stocks: Explicit list of stocks to retain.
        min_stocks: Minimum number of stocks per date required to keep that date.
        min_value: Minimum data value to retain.
        max_value: Maximum data value to retain.
        values: Explicit list of acceptable data values.
        mask_panel_id: Optional Panel identifier providing a boolean mask.
        index_panel_id: Optional Panel identifier supplying the index to keep.
        dropna: If True, drop rows whose values are NaN before persisting.

    Returns:
        JSON string containing the identifier and metadata of the persisted filtered Panel.
    """

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "dates": dates,
        "stocks": stocks,
        "min_stocks": min_stocks,
        "min_value": min_value,
        "max_value": max_value,
        "values": values,
        "mask_panel_id": mask_panel_id,
        "index_panel_id": index_panel_id,
        "dropna": dropna,
    }
    code = _panel_filter_code(panel_id, params)
    return _log_and_execute("Panel_filter", code)


@mcp.tool()
def Panel_log(panel_id: str) -> str:
    """Apply the natural logarithm to a Panel."""
    code = _unary_panel_operation_code("p1.log()", panel_id)
    return _log_and_execute("Panel_log", code)


@mcp.tool()
def Panel_exp(panel_id: str) -> str:
    """Apply the exponential function to a Panel."""
    code = _unary_panel_operation_code("p1.exp()", panel_id)
    return _log_and_execute("Panel_exp", code)


@mcp.tool()
def get_variables_descriptions() -> dict:
    """Return a mapping of Panel identifiers to their descriptions."""
    df = load_variables()
    if "Description" not in df.columns:
        return {}
    return df["Description"].to_dict()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
