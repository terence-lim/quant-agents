# python server.py
from mcp.server.fastmcp import FastMCP
from utils import _log_and_execute, log_message
from qrafti import Panel, panel_or_numeric, str_or_None, numeric_or_None, int_or_None, MEDIA, DATES
from qrafti import winsorize, digitize, characteristics_fill, characteristics_downsample
from qrafti import portfolio_impute, portfolio_weights, cumcount, portfolio_returns, portfolio_metrics
import json
from typing import List

# Create an MCP server
port = 8000
mcp = FastMCP("factor-server", host="0.0.0.0", port=port)

    
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
        str: JSON containing the id of the created Panel with values winsorized to be within the upper and lower bounds
    """
    log_message(tool="Panel_winsorize", code=f"{panel_id=}, {reference_panel_id=}, {lower=}, {upper=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(reference_panel_id, **DATES)
    p3 = p1.apply(winsorize, None if p2 is None else p2, lower=lower, upper=upper).persist()
    log_message(str(p3), "Panel_winsorize", f"{panel_id=}, {reference_panel_id=}, {lower=}, {upper=}")
    return str(p3)


@mcp.tool()
def Panel_quantiles(panel_id: str, cuts: int | list[float], reference_panel_id: str | int | float = '', 
                   ascending: bool = True) -> str:
    """Categorizes panel values based on number of quantiles or specified breakpoints.

    Args:
        panel_id (str): Identifier for the panel data whose first column will be discretized.
        cuts (int | list[float]): Number of quantile-based bins to categorize into, or explicit breakpoints in ``[0, 1]``.
        reference_panel_id (str, optional): Identifier of the indicator panel whose boolean values select
            which rows contribute to the quantile breakpoints. Defaults to using all rows when omitted.
        ascending (bool, optional): If ``True`` (default), lower values receive lower bin labels; otherwise
            the labels are reversed.

    Returns:
        str: the id of the created Panel of quantiles, with the first quantiles numbered 1, in JSON format
    """
    log_message(tool="Panel_digitize", code=f"{panel_id=}, {cuts=}, {reference_panel_id=}, {ascending=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(reference_panel_id, **DATES)
    p3 = p1.apply(digitize, None if p2 is None else p2, cuts=cuts, ascending=ascending).persist()
    log_message(str(p3), "Panel_digitize", f"{panel_id=}, {cuts=}, {reference_panel_id=}, {ascending=}")
    return str(p3)

@mcp.tool()
def Panel_characteristics_fill(panel_ids: List[str]) -> str:
    """If values are not available, then sequentially fill from list of panels in order.

    Args:
        panel_ids (List[str]): Ordered identifiers of panels whose values should be combined. The
            first panel serves as the base and later panels provide replacement data.
    Returns:
        str: JSON string containing the persisted panel identifier for the filled panel.
    """
    log_message(tool="Panel_characteristics_fill", code=f"{panel_ids=}")
    panels = [panel_or_numeric(pid, **DATES) for pid in panel_ids]
    filled = characteristics_fill(*panels, replace=[0]).persist()
    log_message(str(filled), "Panel_characteristics_fill", f"{panel_ids=}")
    return str(filled)

@mcp.tool()
def Panel_characteristics_downsample(panel_id: str, ffill: bool = True, month: int | List[int] | None = None) -> str:
    """Downsample or filters a panel of characteristics by selected months

    Args:
        panel_id (str): Identifier for the source characteristic panel to sample.
        ffill (optional, bool): If True, forward fill values between months.  
          If False, only use values in the specified months.
        month (int | List[int] | None, optional): Single month number or list of month numbers (1-12)
            to filter or downsamples for. When ``None`` (default), samples are generated for every month.

    Returns:
        str: JSON string containing the identifier of the persisted panel of downsampled results.
    """
    log_message(tool="Panel_characteristics_downsample", code=f"{panel_id=}, {month=}")
    characteristics = panel_or_numeric(panel_id, **DATES)
    samples = characteristics_downsample(characteristics, ffill=ffill, month=month).persist()
    log_message(str(samples), "Panel_characteristics_downsample", f"{panel_id=}, {month=}")
    return str(samples)

@mcp.tool()
def Panel_portfolio_returns(port_weights_panel_id: str) -> str:
    """Compute the portfolio or factor returns given factor portfolio weights.
    Args:
        port_weights_panel_id (str): The id of the panel data set for portfolio weights.
    Returns:
        str: JSON format of the id of the created Panel of returns of the factor or portfolio 
    """
    log_message(tool="Panel_portfolio_returns", code=f"{port_weights_panel_id=}")
    p_weights = Panel(str_or_None(port_weights_panel_id), **DATES)
    p_returns = portfolio_returns(p_weights).persist()
    payload = dict(panel_id=str(p_returns.id))
    log_message(str(payload), "Panel_portfolio_returns", f"{port_weights_panel_id=}")
    return json.dumps(payload)


@mcp.tool()
def Panel_portfolio_turnover(weights_panel_id: str) -> str:
    """Compute turnover for a portfolio weights panel using drifted imputation.

    Args:
        weights_panel_id (str): Identifier of the panel containing portfolio weights.

    Returns:
        str: JSON string containing the persisted panel identifier of the computed turnover output.
    """
    log_message(tool="Panel_portfolio_turnover", code=f"{weights_panel_id=}")
    weights = panel_or_numeric(weights_panel_id, **DATES)
    drifted = portfolio_impute(weights, drifted=True)
    delta = weights - drifted
    turnover = delta.apply(np.abs).apply(np.sum, axis=0).apply(np.mean).persist()
    log_message(str(turnover), "Panel_portfolio_turnover", f"{weights_panel_id=}")
    return str(turnover)

@mcp.tool()
def Panel_sequence(panel_id: str) -> str:
    """Compute the cumulative number of available data points for each stock observation.

    Args:
        panel_id (str): Identifier for the panel whose chronological sequence per stock is desired.

    Returns:
        str: JSON string containing the identifier of the persisted panel identifier with cumulative counts.
    """
    log_message(tool="Panel_sequence", code=f"{panel_id=}")
    panel = panel_or_numeric(panel_id, **DATES)
    sequence = panel.trend(cumcount).persist()
    log_message(str(sequence), "Panel_sequence", f"{panel_id=}")
    return str(sequence)


@mcp.tool()
def Panel_portfolios_weights(panel_id: str, other_panel_id: str = None, leverage: float = 1.0, net: bool = True) -> str:
    """Scale the input values to portfolio weights which sum to the given leverage
    Args:
        panel_id (str): The id of the panel data set to convert to portfolio weights.
        other_panel_id (str, optional): The id of another panel data set to indicate rows to include for weights calculation.
        leverage (float): The target leverage for the portfolio weights.
        net (bool): Whether to allow netting of long and short positions.
    """
    log_message(tool="Panel_portfolio_weights", code=f"{panel_id=}, {leverage=}, {net=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = panel_or_numeric(other_panel_id, **DATES)
    p3 = p1.apply(portfolio_weights, None if p2 is None else p2, how='right',
                           leverage=leverage, net=net).persist()
    log_message(str(p3), "Panel_portfolio_weights", f"{panel_id=}, {leverage=}, {net=}")
    return str(p3)

@mcp.tool()
def Panel_performance_metrics(panel_id: str) -> str:
    """
    Compute performance metrics of portfolio returns in a Panel
    Args:
        panel_id (str): The id of the panel data set to evaluate.
    Returns:
        str: Performance evaluation metrics in JSON format.
    """
    log_message(tool="Panel_performance_metrics", code=f"{panel_id=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    summary_dict = portfolio_metrics(p1)
    payload = dict(summary_dict=summary_dict)
    log_message(str(payload), "Panel_performance_metrics", f"{panel_id=}")
    return json.dumps(payload)

@mcp.tool()
def Panel_plot(panel_id: str, other_panel_id: str = '', kind: str ='line', title='') -> str:
    """
    Plot the values of a Panel.
    Optionally, plot it together with another Panel on the same axes.

    Args:
        panel_id (str): The ID of the primary Panel to plot.
        other_panel_id (str, optional): The ID of another Panel to plot together.
        kind: (str, optional): The type of plot to create (e.g., 'line', 'bar', 'scatter').
        title (str, optional): The title of the plot.
    Returns:
        str: Full path name containing the plot image in JSON format.
    """
    log_message(tool="Panel_plot", code=f"{panel_id=}, {other_panel_id=}, {kind=}, {title=}")
    p1 = panel_or_numeric(panel_id, **DATES)
    assert p1.nlevels == 1, f"nlevels of {panel_id}=={p1.nlevels} is not 1"
    p2 = panel_or_numeric(other_panel_id, **DATES)
    if p2:
        assert p2.nlevels == 1, f"nlevels of {other_panel_id}=={p2.nlevels} is not 1"
        fig = p1.plot(p2, kind=kind, title=title)
    else:
        fig = p1.plot(kind=kind, title=title)
    savefig = MEDIA / f"plot_{panel_id}.png"
    fig.savefig(savefig)
    payload = dict(image_path_name='file://' + str(savefig))
    log_message(str(payload), "Panel_plot", f"{panel_id=}, {other_panel_id=}, {kind=}, {title=}")
    return json.dumps(payload)

if __name__ == "__main__":
    # Run with SSE transport
    mcp.run(transport="streamable-http")
