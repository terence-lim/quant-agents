# python server.py
from mcp.server.fastmcp import FastMCP
from qrafti import load_variables
from utils import _log_and_execute, dates_
import json
from typing import Any, List, Optional, Dict, Callable

# Create an MCP server
mcp = FastMCP("metadata-server", host="0.0.0.0", port=8002)

def _binary_panel_operation_code(op: str, panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Return the sandbox code for a binary Panel operation."""
    return f"""
import json
from qrafti import Panel, panel_or_numeric
p1 = panel_or_numeric('{panel_id}', **{dates_})
p2 = panel_or_numeric('{other_panel_id}', **{dates_})
p3 = ({op}).persist()
print(json.dumps({{'result_panel_id': p3.name}}))
"""


def _unary_panel_operation_code(op: str, panel_id: str | int | float) -> str:
    """Return the sandbox code for a unary Panel operation."""
    return f"""
import json
from qrafti import Panel, panel_or_numeric
p1 = panel_or_numeric('{panel_id}', **{dates_})
p2 = ({op}).persist()
print(json.dumps({{'result_panel_id': p2.name}}))
"""


@mcp.tool()
def Panel_matmul(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
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
def Panel_add(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Add two Panels element-wise."""
    code = _binary_panel_operation_code("p1 + p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_add", code)


@mcp.tool()
def Panel_radd(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Add two Panels element-wise (reverse order)."""
    code = _binary_panel_operation_code("p1.__radd__(p2)", panel_id, other_panel_id)
    return _log_and_execute("Panel_radd", code)


@mcp.tool()
def Panel_sub(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Subtract the second Panel from the first Panel."""
    code = _binary_panel_operation_code("p1 - p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_sub", code)


@mcp.tool()
def Panel_rsub(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Subtract the first Panel from the second Panel."""
    code = _binary_panel_operation_code("p1.__rsub__(p2)", panel_id, other_panel_id)
    return _log_and_execute("Panel_rsub", code)


@mcp.tool()
def Panel_mul(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Multiply two Panels element-wise."""
    code = _binary_panel_operation_code("p1 * p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_mul", code)


@mcp.tool()
def Panel_rmul(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Multiply two Panels element-wise (reverse order)."""
    code = _binary_panel_operation_code("p1.__rmul__(p2)", panel_id, other_panel_id)
    return _log_and_execute("Panel_rmul", code)


@mcp.tool()
def Panel_truediv(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Divide the first Panel by the second Panel."""
    code = _binary_panel_operation_code("p1 / p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_truediv", code)


@mcp.tool()
def Panel_rtruediv(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Divide the second Panel by the first Panel."""
    code = _binary_panel_operation_code("p1.__rtruediv__(p2)", panel_id, other_panel_id)
    return _log_and_execute("Panel_rtruediv", code)


@mcp.tool()
def Panel_eq(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Compare two Panels for element-wise equality."""
    code = _binary_panel_operation_code("p1 == p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_eq", code)


@mcp.tool()
def Panel_ne(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Compare two Panels for element-wise inequality."""
    code = _binary_panel_operation_code("p1 != p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_ne", code)


@mcp.tool()
def Panel_lt(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Compare whether the first Panel is less than the second Panel element-wise."""
    code = _binary_panel_operation_code("p1 < p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_lt", code)


@mcp.tool()
def Panel_le(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Compare whether the first Panel is less than or equal to the second Panel element-wise."""
    code = _binary_panel_operation_code("p1 <= p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_le", code)


@mcp.tool()
def Panel_gt(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Compare whether the first Panel is greater than the second Panel element-wise."""
    code = _binary_panel_operation_code("p1 > p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_gt", code)


@mcp.tool()
def Panel_ge(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Compare whether the first Panel is greater than or equal to the second Panel element-wise."""
    code = _binary_panel_operation_code("p1 >= p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_ge", code)


@mcp.tool()
def Panel_or(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Combine two Panels using a logical OR operation."""
    code = _binary_panel_operation_code("p1 | p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_or", code)


@mcp.tool()
def Panel_and(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Combine two Panels using a logical AND operation."""
    code = _binary_panel_operation_code("p1 & p2", panel_id, other_panel_id)
    return _log_and_execute("Panel_and", code)


@mcp.tool()
def Panel_neg(panel_id: str | int | float) -> str:
    """Negate the values of a Panel."""
    code = _unary_panel_operation_code("-p1", panel_id)
    return _log_and_execute("Panel_neg", code)


@mcp.tool()
def Panel_invert(panel_id: str | int | float) -> str:
    """Apply a logical NOT operation to a Panel."""
    code = _unary_panel_operation_code("~p1", panel_id)
    return _log_and_execute("Panel_invert", code)


@mcp.tool()
def Panel_log(panel_id: str | int | float) -> str:
    """Apply the natural logarithm to a Panel."""
    code = _unary_panel_operation_code("p1.log()", panel_id)
    return _log_and_execute("Panel_log", code)


@mcp.tool()
def Panel_exp(panel_id: str | int | float) -> str:
    """Apply the exponential function to a Panel."""
    code = _unary_panel_operation_code("p1.exp()", panel_id)
    return _log_and_execute("Panel_exp", code)


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
    isin: Optional[List[Any]] = None,
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
        isin: Explicit list of acceptable data values.
        mask_panel_id: Optional Panel identifier providing a boolean mask.
        index_panel_id: Optional Panel identifier supplying the index to keep.
        dropna: If True, drop rows whose values are NaN before persisting.

    Returns:
        JSON string containing the identifier
    """

    code = f"""
import json
from qrafti import Panel, panel_or_numeric
p1 = panel_or_numeric('{panel_id}', **{dates_})
mask_panel = panel_or_numeric('{mask_panel_id}', **{dates_})
index_panel = panel_or_numeric('{index_panel_id}', **{dates_})

p2 = p1.filter(start_date={start_date},
    end_date={end_date},
    dates={dates},
    stocks={stocks},
    min_stocks={min_stocks},
    min_value={min_value},
    max_value={max_value},
    isin={isin},
    dropna=True if {dropna} else False,
    mask=mask_panel,
    index=index_panel).persist()
print(json.dumps({{'result_panel_id': p2.name}}))
"""
    return _log_and_execute("Panel_filter", code)

@mcp.tool()
def Panel_shift(panel_id: str, shift: int = 1) -> str:
    """
    Create a new Panel with its date index shifted forward, by one date by default.

    This function moves all data forward in time along the date index.
    The dates with no values to shift from in the dataset are dropped.

    Common use cases include aligning lagged or forward-looking features in 
    time-series and panel data analysis.

    Args:
        panel_id (str): 
            The ID of the existing Panel to shift. This should correspond to 
            a Panel object stored in the cache.
        shift (int, optional): 
            The number of date steps to shift forward. Default is 1.

    Returns:
        str: 
            The ID of the newly created Panel (stored in the cache) whose dates 
            have been shifted forward by one step.
    """
    code = f"""
import json
from qrafti import Panel, panel_or_numeric
p1 = panel_or_numeric('{panel_id}', **{dates_})
p2 = p1.shift(shift={shift}).persist()
print(json.dumps({{'result_panel_id': p2.name}}))
"""
    return _log_and_execute("Panel_shift", code)


SPECIALIZED_AGENT_TOOLS: Dict[str, List[Dict[str, str]]] = {
    "factor_agent_tool": [
        {
            "name": "Panel_characteristics_snapshots",
            "description": "Capture forward-filled characteristic snapshots for selected calendar months.",
        },
        {
            "name": "Panel_characteristics_fill",
            "description": "Fill characteristic panels sequentially using fallback sources and optional value replacements.",
        },
        {
            "name": "Panel_portfolio_turnover",
            "description": "Measure turnover by imputing drifted portfolio weights with optional return inputs.",
        },
        {
            "name": "Panel_sequence",
            "description": "Compute per-stock observation sequence numbers via cumulative counts.",
        },
        {
            "name": "Panel_winsorize",
            "description": "Winsorize panel values using optional reference weights and percentile cutoffs.",
        },
        {
            "name": "Panel_digitize",
            "description": "Discretize panel observations into categories with optional masking and ordering controls.",
        },
        {
            "name": "Panel__portfolio_weights",
            "description": "Construct portfolio weights from raw scores or stock characteristics.",
        },
        {
            "name": "Panel_portfolio_returns",
            "description": "Generate portfolio or factor returns from portfolio weights or factor characteristic panels.",
        },
    ],
    "performance_agent_tool": [
        {
            "name": "Panel_performance_evaluation",
            "description": "Summarize risk and performance statistics for a returns panel.",
        },
        {
            "name": "Panel_plot",
            "description": "Render and persist plots for one or two panels with configurable plot type and title.",
        },
    ],
}

def _summarize_tools(functions: List[Callable[..., str]]) -> List[Dict[str, str]]:
    """Return name/description metadata for the provided MCP tool callables."""

    summaries: List[Dict[str, str]] = []
    for func in functions:
        doc = (func.__doc__ or "").strip()
        description = doc.splitlines()[0] if doc else ""
        summaries.append({"name": func.__name__, "description": description})
    return summaries


def _server_tool_metadata() -> List[Dict[str, str]]:
    """Collect metadata for MCP tools defined in this server module."""

    tool_functions: List[Callable[..., str]] = [
        Panel_matmul,
        Panel_add,
        Panel_radd,
        Panel_sub,
        Panel_rsub,
        Panel_mul,
        Panel_rmul,
        Panel_truediv,
        Panel_rtruediv,
        Panel_eq,
        Panel_ne,
        Panel_lt,
        Panel_le,
        Panel_gt,
        Panel_ge,
        Panel_or,
        Panel_and,
        Panel_neg,
        Panel_invert,
        Panel_filter,
        Panel_log,
        Panel_exp,
        Panel_shift
    ]
    return _summarize_tools(tool_functions)


@mcp.tool()
def get_specialized_agent_tools() -> str:
    """Return specialized and shared MCP tools accessible to planner workflows."""

    payload = {
        "agents": SPECIALIZED_AGENT_TOOLS,
        "server": _server_tool_metadata(),
        "notes": "Each entry lists the tools callable by a specialized agent tool plus shared server MCP tools they may invoke.",
    }
    return json.dumps(payload)

@mcp.tool()
def get_variables_descriptions() -> dict:
    """Return a mapping of Panel identifiers to their descriptions."""
    df = load_variables()
    if "Description" not in df.columns:
        return {}
    return df["Description"].to_dict()

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
