# python research_server.py  
# (c) Terence Lim
from mcp.server.fastmcp import FastMCP
import json
import pandas as pd
from pandas.api.types import is_scalar
from typing import List, Optional
import traceback
import warnings
import logging

from qrafti import Panel, DATES
from rag import RAG
from research_utils import (winsorize, standardize, digitize, characteristics_coalesce, characteristics_resample,
                            portfolio_weights, portfolio_returns, rolling, regression_residuals)
from server_utils import panel_or_numeric, str_or_None, bool_or_None, int_or_None, log_tool, query_rag
from utils import plt_savefig, BENCHMARKS_RAG, CHARACTERISTICS_RAG, JKP_RAG_PATH, CRSP_RAG_PATH
RAG_PATH = CRSP_RAG_PATH  # JKP_RAG_PATH

logging.basicConfig(level=logging.DEBUG)
warnings.simplefilter(action="ignore", category=FutureWarning)

char_rag = RAG(CHARACTERISTICS_RAG, out_dir=RAG_PATH).load()
bench_rag = RAG(BENCHMARKS_RAG, out_dir=RAG_PATH).load()

port = 8000
mcp = FastMCP("research-server", host="0.0.0.0", port=port)

@mcp.tool()
def Panel_binary_op(
    op: str,
    left: str | int | float | bool,
    right: str | int | float | bool,
) -> str:
    """
    Apply a binary operation between two Panel operands (element-wise) or between a Panel and a scalar/boolean.

    How to choose `op`:
      - "add"     : left + right
      - "sub"     : left - right
      - "mul"     : left * right
      - "truediv" : left / right
      - "pow"     : left ** right
      - "eq"      : left == right
      - "ne"      : left != right
      - "lt"      : left < right
      - "le"      : left <= right
      - "gt"      : left > right
      - "ge"      : left >= right
      - "or"      : left | right   (logical/bitwise OR; commonly for boolean Panels/masks)
      - "and"     : left & right   (logical/bitwise AND; commonly for boolean Panels/masks)

    Operands (`left`, `right`):
      - Each operand may be:
          * a Panel identifier (str / int / float, depending on how Panels are keyed in your system), OR
          * a scalar numeric value (int/float) for arithmetic/comparison, OR
          * a boolean (bool) for "and"/"or" operations (or boolean-like masks).
      - Operations are applied element-wise when Panel(s) are involved.

    Args:
        op: Which operation to apply. One of:
            {"add","sub","mul","truediv","pow","eq","ne","lt","le","gt","ge","or","and"}.
        left: Identifier of the left Panel operand, or a scalar/boolean.
        right: Identifier of the right Panel operand, or a scalar/boolean.

    Returns:
        JSON string containing the identifier of the persisted output Panel payload,
        or {"error": "...traceback..."} on failure.
    """
    try:
        p1 = panel_or_numeric(left, **DATES)
        p2 = panel_or_numeric(right, **DATES)

        ops = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "truediv": lambda a, b: a / b,
            "pow": lambda a, b: a ** b,
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "lt": lambda a, b: a < b,
            "le": lambda a, b: a <= b,
            "gt": lambda a, b: a > b,
            "ge": lambda a, b: a >= b,
            "or": lambda a, b: a | b,
            "and": lambda a, b: a & b,
        }

        if op not in ops:
            raise ValueError(
                f"Unsupported op={op!r}. Must be one of {sorted(ops.keys())}."
            )

        p3 = Panel(ops[op](p1, p2))
        out = p3.as_payload()
    except Exception:
        out = dict(error=traceback.format_exc())

    log_tool(
        tool="Panel_binary_op",
        input=dict(op=op, panel_id=left, other_panel_id=right),
        output=out,
    )
    return json.dumps(out)

@mcp.tool()
def Panel_unary_op(
    op: str,
    panel_id: str | int | float | bool,
) -> str:
    """
    Apply a unary operation to a Panel (element-wise) or to a scalar/boolean resolved from `panel_id`.

    How to choose `op`:
      - "neg"   : arithmetic negation, -x
      - "not"   : logical NOT for scalars, bitwise/element-wise invert for Panels
                 (uses `not x` if `is_scalar(x)` else `~x`)
      - "log"   : natural logarithm, x.log()
      - "exp"   : exponential, x.exp()
      - "log1p" : natural log of (1 + x), x.log1p()
      - "expm1" : exp(x) - 1, x.expm1()
      - "abs"   : absolute value, x.abs()
      - "int"   : convert/cast to integer type, x.int()

    Operand (`panel_id`):
      - May be a Panel identifier, or a scalar/boolean.

    Behavior:
      - Operations are applied element-wise when the resolved operand is a Panel-like object.

    Args:
        op: Which unary operation to apply. One of:
            {"neg","not","log","exp","log1p","expm1","abs","int"}.
        panel_id: Identifier of the Panel to operate on, or a scalar/boolean value.

    Returns:
        JSON string containing the identifier of the persisted output Panel payload,
        or {"error": "...traceback..."} on failure.
    """
    try:
        p1 = panel_or_numeric(panel_id, **DATES)

        if op == "neg":
            p3 = Panel(-p1)

        elif op == "not":
            p3 = Panel((not p1) if is_scalar(p1) else (~p1))

        elif op == "log":
            p3 = p1.log()

        elif op == "exp":
            p3 = p1.exp()

        elif op == "log1p":
            p3 = p1.log1p()

        elif op == "expm1":
            p3 = p1.expm1()

        elif op == "abs":
            p3 = p1.abs()

        elif op == "int":
            p3 = p1.int()

        else:
            raise ValueError(
                f"Unsupported op={op!r}. Must be one of "
                f"{['abs','exp','expm1','int','log','log1p','neg','not']}."
            )

        out = p3.as_payload()

    except Exception:
        out = dict(error=traceback.format_exc())

    log_tool(
        tool="Panel_unary_op",
        input=dict(op=op, panel_id=panel_id),
        output=out,
    )
    return json.dumps(out)


@mcp.tool()
def Panel_isin(panel_id: str | int | float, values: List[int]) -> str:
    """
    Create a boolean mask Panel indicating whether each element is contained within a given list of values.
    Args:
        panel_id (str): The id of the panel data to check.
        values (list[int]): A list of values to check for membership.
    Returns:
        JSON string containing the identifier of the persisted Panel with boolean values.
    """
    try:
        p1 = panel_or_numeric(panel_id, **DATES)
        p2 = p1.apply(lambda df: df.isin(values))
        out = p2.as_payload()
    except Exception as e:
        out = dict(error=traceback.format_exc())
    log_tool(tool="Panel_isin",
             input=dict(panel_id=panel_id, values=values),
             output=out)
    return json.dumps(out)

@mcp.tool()
def Panel_restrict(panel_id: str, mask_panel_id: Optional[str] = None,
                   subset_panel_id: Optional[str] = None, positive: bool = False) -> str:
    """Restrict a Panel by keeping only samples that satisfy the optional criteria.

    Args:
        panel_id: Identifier of the Panel to filter.
        mask_panel_id: Optional Panel identifier providing a boolean mask to keep rows that are True.
        subset_panel_id: Optional Panel identifier supplying the subset to keep rows whose index is in the subset.
        positive: Optional bool indicating whether to restrict to positive values only
    Returns:
        JSON string containing the identifier of the persisted filtered Panel.
    """
    try:
        p1 = panel_or_numeric(panel_id, **DATES)
        mask_panel = panel_or_numeric(mask_panel_id, **DATES)
        subset_panel = panel_or_numeric(subset_panel_id, **DATES)
        kwargs = dict(min_value=1e-6) if bool_or_None(positive) else {}
        p2 = p1.restrict(mask=mask_panel, subset=subset_panel, **kwargs)
        out = p2.as_payload()
    except Exception as e:
        out = dict(error=traceback.format_exc())
    log_tool(tool="Panel_restrict", 
             input=dict(panel_id=panel_id, mask_panel_id=mask_panel_id, subset_panel_id=subset_panel_id, **kwargs),
             output=out)
    return json.dumps(out)

@mcp.tool()
def Panel_lag(panel_id: str, months: int = 1) -> str:
    """
    Tool: Shift (lag/lead) a cached Panel by relabeling its date index by whole months.

    Use this tool when the intent is a **pure time shift** of existing rows:
    - You want the same observations, but with their dates moved forward/backward by `months`.
    - You do **not** want to change frequency, align to month-ends, aggregate, or fill missing values.
    - It does not create new rows; it only relabels dates for all rows.

    When NOT to use
    ---------------
    - Do not use for downsampling to selected months or month-ends.

    Parameters
    ----------
    panel_id : str
        Identifier of the existing cached Panel to shift.

    months : int, default 1
        Number of months to shift the date index:
        - Positive values shift dates forward (lag the data relative to time).
        - Negative values shift dates backward (lead the data relative to time).

    Returns
    -------
    str
        JSON string payload for the persisted shifted Panel (including its identifier),
        or an error payload if the operation fails.
    """
    try:
        p1 = panel_or_numeric(panel_id, **DATES)
        months = int_or_None(months)
        p2 = p1.shift(shift=months)
        out = p2.as_payload()
    except Exception as e:
        out = dict(error=traceback.format_exc())
    log_tool(tool="Panel_lag", 
             input=dict(panel_id=panel_id, months=months),
             output=out)
    return json.dumps(out)

@mcp.tool()
def Panel_lookup(query: str, panel_type: str) -> dict:
    """
    Lookup Panel identifiers and descriptions using a natural-language query or an ID-like string.

    When to call this tool:
      - You have a partial/approximate Panel name/description and want the best matches.
      - You have something that looks like an ID and want to retrieve the closest matching entries.

    How to choose `panel_type`:
      - "characteristics": search stock characteristics Panels (signals/factors/attributes).
      - "benchmarks": search benchmark portfolio return Panels (index/benchmark return series).

    Args:
        query: Keywords / phrase / ID-like string describing what you want to find.
        panel_type: One of {"characteristics","benchmarks"} controlling which catalog(s) to search.

    Returns:
        A dict of best matches (top 10 per catalog searched). The structure mirrors the underlying
        RAG result format and will contain Panel IDs and their descriptions. On failure, returns:
            {"error": "<traceback>"}.
    """
    try:
        if panel_type not in {"characteristics", "benchmarks"}:
            raise ValueError(
                f"Invalid panel_type={panel_type!r}. Must be one of "
                f"{{'characteristics','benchmarks'}}."
            )
        out = query_rag(query, rag=char_rag if panel_type in {"characteristics"} else bench_rag, top_n=10)

    except Exception:
        out = dict(error=traceback.format_exc())

    log_tool(
        tool="Panel_lookup",
        input=dict(query=query, panel_type=panel_type),
        output=out,
    )
    return out

@mcp.tool()
def Panel_standardize(panel_id: str, reference_panel_id: str = '') -> str:
    """
    Create a Panel that standardizes the values of the given panel data.
    Args:
        panel_id (str): The id of the panel data set to standardize.
        reference_panel_id (str, optional): The id of the reference panel data set which indicates which
            rows of the panel data set to use for computing the mean and standard deviation.
            If not provided, the mean and standard deviation will be computed based on all the values in the panel_id data set.
    Returns:
        JSON string containing the identifier of the persisted standardized Panel.
    """
    try:
        p1 = panel_or_numeric(panel_id, **DATES)
        p2 = panel_or_numeric(reference_panel_id, **DATES)
        p3 = p1.apply(standardize, 1 if p2 is None else p2)  # if p2 is None, then use all
        out = p3.as_payload()
    except Exception as e:
        out = dict(error=traceback.format_exc())
    log_tool(tool="Panel_standardize", 
             input=dict(panel_id=panel_id, reference_panel_id=reference_panel_id),
             output=out)
    return json.dumps(out)

@mcp.tool()
def Panel_winsorize(panel_id: str, reference_panel_id: str = '', lower: float = 0.0, upper: float = 1.0) -> str:
    """
    Create a Panel that winsorizes the values of the given panel data.
    Args:
        panel_id (str): The id of the panel data set to winsorize.
        upper (float): The upper percentile to winsorize to (between 0 and 1).
        lower (float): The lower percentile to winsorize to (between 0 and 1).
        reference_panel_id (str, optional): The id of the reference panel data set which indicates which
            rows of the panel data set to use for computing the upper and lower bounds.
            If not provided, the winsorization bounds will be computed based on all the values in the panel_id data set.
    Returns:
        JSON string containing the identifier of the persisted winsorized Panel.
    """
    try:
        p1 = panel_or_numeric(panel_id, **DATES)
        p2 = panel_or_numeric(reference_panel_id, **DATES)
        p3 = p1.apply(winsorize, 1 if p2 is None else p2, lower=lower, upper=upper)
        out = p3.as_payload()
    except Exception as e:
        out = dict(error=traceback.format_exc())
    log_tool(tool="Panel_winsorize", 
             input=dict(panel_id=panel_id, reference_panel_id=reference_panel_id, lower=lower, upper=upper),
             output=out)
    return json.dumps(out)

@mcp.tool()
def Panel_quantiles(panel_id: str, cuts: int | list[float], reference_panel_id: str | int | float = '', 
                   ascending: bool = True) -> str:
    """
    Categorizes or discretizes Panel values into ranked bins (e.g., deciles, quintiles) 
    based on quantile thresholds.

    ### Key Operations:
    * **Equal-Spaced Quantiles**: If `cuts` is an integer (e.g., 5), it creates N equal-sized 
      bins (quintiles).
    * **Specific Breakpoints**: If `cuts` is a list (e.g., [0.3, 0.7]), it uses those 
      probability thresholds to define bin boundaries.
    * **Reference-Based Scaling**: If a `reference_panel_id` is provided, the quantile 
      breakpoints are calculated using ONLY the subset of data where the reference panel 
      is True (e.g., defining deciles based only on 'Large Cap' stocks), but these bins 
      are then applied to the entire dataset.

    Args:
        panel_id (str): The identifier for the Panel data to be discretized.
        cuts (int | list[float]): 
            - Use an `int` for the number of bins (e.g., 10 for deciles).
            - Use a `list` for specific internal probabilities (e.g., [0.33, 0.66] for 3 bins).
        reference_panel_id (str, optional): ID of a boolean/indicator Panel. If provided, 
            only True rows in this panel define the distribution thresholds. Defaults to 
            using the whole universe.
        ascending (bool, optional): 
            - `True` (default): Bin 1 = Lowest values (standard ranking).
            - `False`: Bin 1 = Highest values (useful for "short" factors).

    Returns:
        str: JSON containing the new `panel_id`. The values in this panel will be 
             integers starting from 1 up to the number of bins.    
    """
    try:
        p1 = panel_or_numeric(panel_id, **DATES)
        p2 = panel_or_numeric(reference_panel_id, **DATES)
        p3 = p1.apply(digitize, 1 if p2 is None else p2, cuts=cuts, ascending=ascending)
        out = p3.as_payload()
    except Exception as e:
        out = dict(error=traceback.format_exc())
    log_tool(tool="Panel_quantiles",
             input=dict(panel_id=panel_id, cuts=cuts, reference_panel_id=reference_panel_id, ascending=ascending),
             output=out)    
    return json.dumps(out)

@mcp.tool()
def Panel_characteristics_coalesce(panel_ids: List[str]) -> str:
    """If values are not available, then sequentially fill from list of panels in order.
    Args:
        panel_ids (List[str]): Ordered identifiers of panels whose values should be combined. The
            first panel serves as the base and later panels provide replacement data.
    Returns:
        JSON string containing the persisted panel identifier for the filled panel.
    """
    try:
        panels = [panel_or_numeric(pid, **DATES) for pid in panel_ids]
        filled = characteristics_coalesce(*panels, replace=[0])
        out = filled.as_payload()
    except Exception as e:
        out = dict(error=traceback.format_exc())
    log_tool(tool="Panel_characteristics_coalesce",
             input=dict(panel_ids=panel_ids),
             output=out)
    return json.dumps(out)


@mcp.tool()
def Panel_characteristics_resample(
    panel_id: str,
    ffill: bool = True,
    month: int | List[int] | None = None
) -> str:
    """
    Tool: Resample a cached characteristics Panel to lower-frequency target months,
    optionally forward-filling each entity's last-known value to each target date.

    Use this tool when the intent is **frequency selection / downsampling**:
    - You want observations aligned to a set of *target sampling dates* determined by `month`.
    - You may want missing values at the target date filled using the latest observation since the
      previous target date (`ffill=True`).
    - This tool chooses target months and (optionally) carries forward values
      so each (entity, sampled_date) reflects the latest available characteristic within the window.

    Typical use cases
    -----------------
    - Convert irregular/daily characteristics into a monthly (or selected-month) panel.
    - Ensure each sampled month has one value per entity, using last observation carried forward.

    Parameters
    ----------
    panel_id : str
        Identifier of the cached characteristics Panel to resample. Expected to be indexed by
        (date, stock_id).

    ffill : bool, default True
        Forward-fill behavior within each sampling window:
        - True: for each target date, use the latest observation for each entity observed after the
          previous target date and on/before the target date.
        - False: include only observations that occur exactly on the target date.

    month : int | List[int] | None, default None
        Target sampling months:
        - None: generate samples for every month in the date range.
        - int or list[int] (1..12): sample only those months (e.g., [3, 9] for Mar/Sep).

    Returns
    -------
    str
        JSON string payload for the persisted resampled Panel (including its identifier),
        or an error payload if the operation fails.
    """
    try:
        characteristics = panel_or_numeric(panel_id, **DATES)
        samples = characteristics_resample(characteristics, ffill=ffill, month=month)
        out = samples.as_payload()
    except Exception as e:
        out = dict(error=traceback.format_exc())
    log_tool(tool="Panel_characteristics_resample", 
             input=dict(panel_id=panel_id, ffill=ffill, month=month),
             output=out)
    return json.dumps(out)

@mcp.tool()
def Panel_annual_change(panel_id: str, op: str = "pct", quarterly: bool = False) -> str:
    """
    Compute annual change from a Panel that contains either annual or quarterly observations.

    How to choose `op`:
      - "pct"  : annual percent change (uses pandas.DataFrame.pct_change)
                - For annual data: compares t vs t-1 year (interval=12, periods=1)
                - For quarterly data: compares t vs t-4 quarters (interval=3, periods=4)
      - "diff" : annual difference (uses pandas.DataFrame.diff)
                - For annual data: value(t) - value(t-1 year) (interval=12, periods=1)
                - For quarterly data: value(t) - value(t-4 quarters) (interval=3, periods=4)

    How `quarterly` works:
      - quarterly=False (default): input Panel is annual frequency (one observation per year).
      - quarterly=True: input Panel is quarterly frequency (one observation per quarter).

    Args:
        panel_id: Identifier for the source Panel to compute annual changes from.
        op: Which annual change to compute. One of {"pct","diff"}.
        quarterly: Whether the input Panel is quarterly (True) or annual (False).

    Returns:
        JSON string containing the identifier (and any other fields) of the persisted output Panel payload,
        or {"error": "...traceback..."} on failure.
    """
    try:
        p1 = panel_or_numeric(panel_id, **DATES)

        if op not in {"pct", "diff"}:
            raise ValueError("op must be either 'pct' or 'diff'")

        func = pd.DataFrame.pct_change if op == "pct" else pd.DataFrame.diff

        if quarterly:
            p2 = p1.trend(func, interval=3, periods=4)
        else:
            p2 = p1.trend(func, interval=12, periods=1)

        out = p2.as_payload()
    except Exception:
        out = dict(error=traceback.format_exc())

    log_tool(
        tool="Panel_annual_change",
        input=dict(panel_id=panel_id, op=op, quarterly=quarterly),
        output=out,
    )
    return json.dumps(out)

@mcp.tool()
def Panel_rolling(panel_id: str, window: int, skip: int = 0, agg: str = "sum", interval: int = 1) -> str:
    """Computes rolling aggregates over past `window` periods of `interval` length, skipping most recent `skip` periods;
    useful for measuring momentum, volatility, or other time-aggregated statistic.

    Args:
        panel_id (str): Identifier for the source characteristic panel to compute percent change.
        window (int): Number of past periods, where each period is of length `interval` number of months,
           over which to compute the rolling aggregate statistic.
        skip (int): Number of the most recent periods to skip in computing the rolling statistic.
        agg (str): Statistic to aggregate, e.g. "sum" (default), "mean", "max", "min", "std",
        interval (int): Length of each interval in number of months, e.g. 12 for annual,
           3 for quarterly, 1 (default) for monthly.

    Use cases:
    - To compute price momentum over the past 12 months, skipping the most recent month, 
      use window=12, skip=1, interval=1, agg="sum" on a panel of the natural logarithm of one plus (log1p)
      monthly stock returns. then computing the exponential of the sum minus 1 (expm1).

    - To compute volatility over, for example, the past 3 years, skipping the most recent year, 
      use window=3, skip=1, interval=12, agg="mean" on a panel of squared monthly stock returns.

    Returns:
        JSON string containing the identifier of the persisted panel of rolling statistics.
    """
    try:
        p1 = panel_or_numeric(panel_id, **DATES)
        p2 = p1.trend(rolling, window=window, skip=skip, agg=agg, interval=interval)
        p3 = p2
        out = p3.as_payload()
    except Exception as e:
        out = dict(error=traceback.format_exc())
    log_tool(tool="Panel_rolling",
             input=dict(panel_id=panel_id, window=window, skip=skip, agg=agg, interval=interval),
             output=out)    
    return json.dumps(out)

@mcp.tool()
def Panel_portfolio_returns(panel_id: str) -> str:
    """
    Calculates the realized time-series returns for a portfolio based on specific weights. 
    
    ### Mandatory Workflow Context
    This tool is the **required second step** after generating portfolio weights. 
    If you have just used `Panel_portfolio_weights` or any tool that produces a `panel_id` 
    containing weights, you MUST call this tool next to determine how that portfolio 
    actually performed over time.

    ### Functionality
    It transforms a panel of asset weights into a single panel of portfolio returns 
    by calculating the weighted average of underlying asset returns:
    $$R_p = \\sum_{i=1}^{n} w_i r_i$$

    Args:
        panel_id (str): The unique ID of the panel containing the asset weights (typically 
                        the output from a weight-construction tool).
    Returns:
        str: A JSON payload containing the new `panel_id` for the resulting returns series.

    Note:
        If portfolio stock weights are missing on any month-end dates, 
        they will implicitly be imputed by drifting the prior month's stock weights.
    """ 
    try:
        p_weights = Panel().load(str_or_None(panel_id), **DATES)
        p_returns = portfolio_returns(p_weights)
        out = p_returns.as_payload()
    except Exception as e:
        out = dict(error=traceback.format_exc())
    log_tool(tool="Panel_portfolio_returns", 
             input=dict(panel_id=panel_id),
             output=out)
    return json.dumps(out)

@mcp.tool()
def Panel_portfolio_weights(panel_id: str, other_panel_id: str = None) -> str:
    """
    Constructs normalized portfolio weights that sum to 1.0. 
    
    Use this tool ALWAYS when asked to:
    1. Create a stock portfolio from raw values like market capitalization.
    2. Normalize stock weights so the total allocation equals 100% (1.0).
    3. Given a selection mask (boolean indicators) from a set of stock characteristics.
    
    This tool is a mandatory prerequisite before calling 'Panel_portfolio_returns'.

    Args:
        panel_id (str): The 'Selection Mask'. A panel of boolean indicators (True/False or 1/0) 
                        identifying which stocks to include in the portfolio for each period.
        other_panel_id (str, optional): The 'Raw Weights'. A panel containing the values used 
                        to calculate proportional weights (e.g., Market Cap, Book Value, or Alpha scores). 
                        These values will be scaled so that the included stocks sum to 1.0. If None,
                        then the stocks will be equal-weighted in the portfolio
    Returns:
        str: A JSON payload containing the new `panel_id` for the resulting panel of portfolio weights
    """
    try:
        p1 = panel_or_numeric(panel_id, **DATES)
        p2 = panel_or_numeric(other_panel_id, **DATES)
        p3 = p2.apply(portfolio_weights, p1, fill_value=0)
        out = p3.as_payload()
    except Exception as e:
        out = dict(error=traceback.format_exc())
    log_tool(tool="Panel_portfolio_weights", 
             input=dict(panel_id=panel_id, other_panel_id=other_panel_id),
             output=out)
    return json.dumps(out)

@mcp.tool()
def Panel_plot(panel_id: str, other_panel_id: str = '', kind: str ='line', title='') -> str:
    """Plot the values of a Panel, optionally with another Panel.
    Args:
        panel_id (str): The ID of the primary Panel to plot.
        other_panel_id (str, optional): The ID of another Panel to plot together.
        kind: (str, optional): The type of plot to create (e.g., 'line', 'bar', 'scatter').
        title (str, optional): The title of the plot.
    Returns:
        str: A JSON payload containing the image file name to be reported to the user.
    """
    try:
        p1 = panel_or_numeric(panel_id, **DATES)
        p2 = panel_or_numeric(other_panel_id, **DATES)
        if p2:
            ax = p1.plot(p2, kind=kind, title=title)
        else:
            ax = p1.plot(kind=kind, title=title)
        savefig = plt_savefig()
        out = dict(image_path_name=str(savefig))
    except Exception as e:
        out = dict(error=traceback.format_exc())
    log_tool(tool="Panel_plot", 
             input=dict(panel_id=panel_id, other_panel_id=other_panel_id, kind=kind, title=title),
             output=out)
    return json.dumps(out)


if __name__ == "__main__":
    # Run with SSE transport
    mcp.run(transport="streamable-http")
