# python server.py
from mcp.server.fastmcp import FastMCP
from utils import _log_and_execute, dates_

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
    code = f"""
import json
from qrafti import Panel, winsorize, panel_or_numeric
p1 = panel_or_numeric('{panel_id}', **{dates_})
p2 = panel_or_numeric('{reference_panel_id}', **{dates_})
p3 = p1.apply(winsorize, None if p2 is None else p2, lower={lower}, upper={upper}).persist()
print(str(p3))
"""
    return _log_and_execute('Panel_winsorize' , code)


@mcp.tool()
def Panel_digitize(panel_id: str, cuts: int | list[float], reference_panel_id: str | int | float = '', 
                   ascending: bool = True) -> str:
    """Discretize panel values into categories based on number of quantiles or specified breakpoints.

    Args:
        panel_id (str): Identifier for the panel data whose first column will be discretized.
        cuts (int | list[float]): Number of quantile-based bins to categorize into, or explicit breakpoints in ``[0, 1]``.
        reference_panel_id (str, optional): Identifier of the indicator panel whose boolean values select
            which rows contribute to the quantile breakpoints. Defaults to using all rows when omitted.
        ascending (bool, optional): If ``True`` (default), lower values receive lower bin labels; otherwise
            the labels are reversed.

    Returns:
        str: the id of the created Panel of categories, with the first category numbered 1, in JSON format
    """
    code = f"""
import json
from qrafti import Panel, digitize, panel_or_numeric
p1 = panel_or_numeric('{panel_id}', **{dates_})
p2 = panel_or_numeric('{reference_panel_id}', **{dates_})
p3 = p1.apply(digitize, None if p2 is None else p2, cuts={cuts}, ascending={ascending}).persist()
print(str(p3))
"""
    return _log_and_execute('Panel_digitize', code)

@mcp.tool()
def Panel_characteristics_fill(panel_ids: list[str]) -> str:
    """If values are not available, then sequentially fill from list of panels in order.

    Args:
        panel_ids (list[str]): Ordered identifiers of panels whose values should be combined. The
            first panel serves as the base and later panels provide replacement data.
    Returns:
        str: JSON string containing the persisted panel identifier for the filled panel.
    """
    code = f"""
import json
from qrafti import Panel, characteristics_fill, panel_or_numeric
panels = [panel_or_numeric(pid, **{dates_}) for pid in {panel_ids}]
filled = characteristics_fill(*panels, replace=[0]).persist()
print(str(filled))
"""
    return _log_and_execute('Panel_characteristics_fill', code)

@mcp.tool()
def Panel_characteristics_downsample(panel_id: str, month: int | list[int] | None = None) -> str:
    """Downsample or filters a panel of characteristics by selected months

    Args:
        panel_id (str): Identifier for the source characteristic panel to sample.
        month (int | list[int] | None, optional): Single month number or list of month numbers (1-12)
            to filter or downsamples for. When ``None`` (default), samples are generated for every month.

    Returns:
        str: JSON string containing the identifier of the persisted panel of downsampled results.
    """
    code = f"""
import json
from qrafti import Panel, characteristics_downsample, panel_or_numeric
characteristics = panel_or_numeric('{panel_id}', **{dates_})
samples = characteristics_downsample(characteristics, month={month}).persist()
print(str(samples))
"""
    return _log_and_execute('Panel_characteristics_downsample', code)

@mcp.tool()
def Panel_portfolio_turnover(weights_panel_id: str) -> str:
    """Compute turnover for a portfolio weights panel using drifted imputation.

    Args:
        weights_panel_id (str): Identifier of the panel containing portfolio weights.

    Returns:
        str: JSON string containing the persisted panel identifier of the computed turnover output.
    """

    code = f"""
import json
import numpy as np
from qrafti import Panel, portfolio_impute, panel_or_numeric
weights = panel_or_numeric('{weights_panel_id}', **{dates_})
drifted = portfolio_impute(weights, drifted=True)
delta = weights - drifted
turnover = delta.apply(np.abs).apply(np.sum, axis=0).apply(np.mean).persist()
print(str(turnover))
"""

    return _log_and_execute('Panel_portfolio_turnover', code)


@mcp.tool()
def Panel_sequence(panel_id: str) -> str:
    """Compute the cumulative number of available data points for each stock observation.

    Args:
        panel_id (str): Identifier for the panel whose chronological sequence per stock is desired.

    Returns:
        str: JSON string containing the identifier of the persisted panel identifier with cumulative counts.
    """
    code = f"""
import json
from qrafti import Panel, cumcount, panel_or_numeric

panel = Panel('{panel_id}', **{dates_})
sequence = panel.trend(cumcount).persist()
print(str(sequence))
"""
    return _log_and_execute('Panel_sequence', code)


@mcp.tool()
def Panel_portfolios_weights(panel_id: str, leverage: float = 1.0, net: bool = True) -> str:
    """Scale the input values to portfolio weights which sum to the given leverage
    Arguments:
        x: DataFrame with initial portfolio values
        leverage: Total leverage to scale the final portfolio weights to
        net: If False, scale the average of the sum of absolute long and sum of absolute short weights to the leverage.  
             If True (default), scale the absolute sum of weights to the leverage. 
    Returns:
        pd.Series with the scaled portfolio weights
    Usage:
        panel_frame.apply(portfolio_weights, leverage=leverage, net=False)
    """
    code = f"""
import json
from qrafti import Panel, portfolio_weights, panel_or_numeric
p1 = panel_or_numeric('{panel_id}', **{dates_})
p2 = p1.apply(portfolio_weights, net={net}, leverage={leverage}).persist()
print(str(p2))
"""
    return _log_and_execute('Panel_portfolio_weights', code)

@mcp.tool()
def Panel_portfolio_returns(port_weights_panel_id: str) -> str:
    """Compute the portfolio or factor returns given factor portfolio weights.
    Args:
        port_weights_panel_id (str): The id of the panel data set for portfolio weights.
    Returns:
        str: JSON format of the id of the created Panel of returns of the factor or portfolio 
    """
    code = f"""
import json
from qrafti import Panel, portfolio_returns, panel_or_numeric
p_weights = panel_or_numeric('{port_weights_panel_id}', **{dates_})
p_returns = portfolio_returns(p_weights).persist()
print(str(p_returns))
"""
    return _log_and_execute('Panel_portfolio_returns', code)


if __name__ == "__main__":
    # Run with SSE transport
    mcp.run(transport="streamable-http")
