# python server.py
#
# TO DO 11/5: Take out get_specialized_tools()
# - the plan only needs to be general enough to delegate to agent
# - and the planner should be prompted to provide as much of the query as possible, so no info is lost
#
from mcp.server.fastmcp import FastMCP
from qrafti import load_variables
from utils import _log_and_execute, log_message
import json
from typing import Any, List, Optional, Dict, Callable

import json
from qrafti import Panel, panel_or_numeric, str_or_None, numeric_or_None, int_or_None, DATES

#import logging
#logging.basicConfig(level=logging.DEBUG)

# Create an MCP server
mcp = FastMCP("metadata-server", host="0.0.0.0", port=8002)

# @mcp.tool()
# def Panel_matmul(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
#    """Compute the dot product (matrix multiplication) between two Panels.
#    Args:
#        panel_id (str): The id of the first panel data set.
#        other_panel_id (str): The id of the second panel data set.
#    Returns:
#        str: the id of the created Panel in the cache in JSON format
#    """
#    code = _binary_panel_operation_code("p1 @ p2", panel_id, other_panel_id)
#    return _log_and_execute("Panel_matmul", code)


@mcp.tool()
def Panel_add(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Add two Panels element-wise, or adds first portfolio's weights or returns to the second portfolio"""
    log_message(tool="Panel_add", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p1 + p2).persist()
    log_message(str(p3), "Panel_add", f"{panel_id=}, {other_panel_id=}")
    return str(p3)


@mcp.tool()
def Panel_radd(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Add two Panels element-wise (reverse order), or adds second portfolio's weights or returns to the first"""
    log_message(tool="Panel_radd", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p2 + p1).persist()
    log_message(str(p3), "Panel_radd", f"{panel_id=}, {other_panel_id=}")
    return str(p3)


@mcp.tool()
def Panel_sub(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Subtract the second Panel from the first Panel, or combines first portfolio's weights or returns to the negative of the second"""
    log_message(tool="Panel_sub", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p1 - p2).persist()
    log_message(str(p3), "Panel_sub", f"{panel_id=}, {other_panel_id=}")
    return str(p3)  


@mcp.tool()
def Panel_rsub(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Subtract the first Panel from the second Panel, or combines second portfolio's weights or returns to the negative of the first"""
    log_message(tool="Panel_rsub", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p2 - p1).persist()
    log_message(str(p3), "Panel_rsub", f"{panel_id=}, {other_panel_id=}")
    return str(p3)


@mcp.tool()
def Panel_mul(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Multiply two Panels element-wise, or multiples first portfolio's weights or returns by the other int or float value"""
    log_message(tool="Panel_mul", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p1 * p2).persist()
    log_message(str(p3), "Panel_mul", f"{panel_id=}, {other_panel_id=}")
    return str(p3)


@mcp.tool()
def Panel_rmul(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Multiply two Panels element-wise (reverse order)."""
    log_message(tool="Panel_rmul", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p2 * p1).persist()
    log_message(str(p3), "Panel_rmul", f"{panel_id=}, {other_panel_id=}")
    return str(p3)  


@mcp.tool()
def Panel_truediv(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Divide the first Panel by the second Panel, or divides first portfolio's weights or returns by the other int or float value"""
    log_message(tool="Panel_truediv", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p1 / p2).persist()
    log_message(str(p3), "Panel_truediv", f"{panel_id=}, {other_panel_id=}")
    return str(p3)


@mcp.tool()
def Panel_rtruediv(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Divide the second Panel by the first Panel."""
    log_message(tool="Panel_rtruediv", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p2 / p1).persist()
    log_message(str(p3), "Panel_rtruediv", f"{panel_id=}, {other_panel_id=}")
    return str(p3)


@mcp.tool()
def Panel_eq(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Compare two Panels for element-wise equality."""
    log_message(tool="Panel_eq", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p1 == p2).persist()
    log_message(str(p3), "Panel_eq", f"{panel_id=}, {other_panel_id=}")
    return str(p3)


@mcp.tool()
def Panel_ne(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Compare two Panels for element-wise inequality."""
    log_message(tool="Panel_ne", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p1 != p2).persist()
    log_message(str(p3), "Panel_ne", f"{panel_id=}, {other_panel_id=}")
    return str(p3)  


@mcp.tool()
def Panel_lt(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Compare whether the first Panel is less than the second Panel element-wise."""
    log_message(tool="Panel_lt", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p1 < p2).persist()
    log_message(str(p3), "Panel_lt", f"{panel_id=}, {other_panel_id=}")
    return str(p3)


@mcp.tool()
def Panel_le(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Compare whether the first Panel is less than or equal to the second Panel element-wise."""
    log_message(tool="Panel_le", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p1 <= p2).persist()
    log_message(str(p3), "Panel_le", f"{panel_id=}, {other_panel_id=}")
    return str(p3)


@mcp.tool()
def Panel_gt(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Compare whether the first Panel is greater than the second Panel element-wise."""
    log_message(tool="Panel_gt", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p1 > p2).persist()
    log_message(str(p3), "Panel_gt", f"{panel_id=}, {other_panel_id=}")
    return str(p3)


@mcp.tool()
def Panel_ge(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Compare whether the first Panel is greater than or equal to the second Panel element-wise."""
    log_message(tool="Panel_ge", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p1 >= p2).persist()
    log_message(str(p3), "Panel_ge", f"{panel_id=}, {other_panel_id=}")
    return str(p3)  


@mcp.tool()
def Panel_or(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Combine two Panels using a logical OR operation."""
    log_message(tool="Panel_or", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p1 | p2).persist()
    log_message(str(p3), "Panel_or", f"{panel_id=}, {other_panel_id=}")
    return str(p3)

@mcp.tool()
def Panel_and(panel_id: str | int | float, other_panel_id: str | int | float) -> str:
    """Combine two Panels using a logical AND operation."""
    log_message(tool="Panel_and", code=f"{panel_id=}, {other_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = (p1 & p2).persist()
    log_message(str(p3), "Panel_and", f"{panel_id=}, {other_panel_id=}")
    return str(p3)


@mcp.tool()
def Panel_neg(panel_id: str | int | float) -> str:
    """Negate the values of a Panel."""
    log_message(tool="Panel_neg", code=f"{panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = -p1
    log_message(str(p2), "Panel_neg", f"{panel_id=}")
    return str(p2)


@mcp.tool()
def Panel_invert(panel_id: str | int | float) -> str:
    """Apply a logical NOT operation to a Panel."""
    log_message(tool="Panel_invert", code=f"{panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = ~p1
    log_message(str(p2), "Panel_invert", f"{panel_id=}")
    return str(p2)


@mcp.tool()
def Panel_log(panel_id: str | int | float) -> str:
    """Apply the natural logarithm to a Panel."""
    log_message(tool="Panel_log", code=f"{panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = p1.log().persist()
    log_message(str(p2), "Panel_log", f"{panel_id=}")
    return str(p2)

@mcp.tool()
def Panel_exp(panel_id: str | int | float) -> str:
    """Apply the exponential function to a Panel."""
    log_message(tool="Panel_exp", code=f"{panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = p1.exp().persist()
    log_message(str(p2), "Panel_exp", f"{panel_id=}")
    return str(p2)

@mcp.tool()
def Panel_isin(panel_id: str | int | float, values: List[str]) -> str:
    """
    Create a boolean mask Panel indicating whether each element is contained within a given list of values.
    Args:
        panel_id (str): The id of the panel data to check.
        values (list[str]): A list of values to check for membership.
    Returns:
        str: the id of the created Panel in the cache in JSON format
    """
    log_message(tool="Panel_isin", code=f"{panel_id=}, {values=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = p1.apply(lambda df: df.isin(values)).persist()
    log_message(str(p2), "Panel_isin", f"{panel_id=}, {values=}")
    return str(p2)
    
@mcp.tool()
def Panel_filter(panel_id: str, mask_panel_id: Optional[str] = None, index_panel_id: Optional[str] = None,
) -> str:
    """Filter a Panel by keeping only rows that satisfy the specify the optional criteria.

    Args:
        panel_id: Identifier of the Panel to filter.
        mask_panel_id: Optional Panel identifier providing a boolean mask.
        index_panel_id: Optional Panel identifier supplying the index to keep.

    Returns:
        JSON string containing the identifier
    """
    log_message(tool="Panel_filter", code=f"{panel_id=}, {mask_panel_id=}, {index_panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    mask_panel = panel_or_numeric(mask_panel_id, **DATES)
    index_panel = panel_or_numeric(index_panel_id, **DATES)
    p2 = p1.filter(mask=mask_panel, index=index_panel).persist()
    log_message(str(p2), "Panel_filter", f"{mask_panel_id=}, {index_panel_id=}")
    return str(p2)

@mcp.tool()
def Panel_shift(panel_id: str, months: int = 1) -> str:
    """
    Shift the date index of a Panel by a specified number of months, effectively lagging the data.

    Args:
        panel_id (str): 
            The ID of the existing Panel to shift. This should correspond to 
            a Panel object stored in the cache.
        months (int, optional):
            The number of months to shift the date index forward. Default is 1.
            A negative number means to shift the data index backward.

    Returns:
        str: 
            The ID of the newly created Panel (stored in the cache) whose dates 
            have been shifted forward.
    """
    log_message(output=f"Shifting Panel {panel_id} by {months} months", tool="Panel_shift", code=f"{panel_id=}, {months=}") 
    p1 = panel_or_numeric(panel_id, **DATES)
    months = int_or_None(months)
    p2 = p1.shift(shift=months).persist()
    output = str(p2)
    log_message(output=output, tool="Panel_shift", code=f"{panel_id=}, {months=}")
    return output


@mcp.tool()
def get_variables_descriptions() -> dict:
    """Return a mapping of Panel identifiers to their descriptions."""
    df = load_variables()
    if "Description" not in df.columns:
        return {}
    return df["Description"].to_dict()

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
