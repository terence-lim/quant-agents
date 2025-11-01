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
print(json.dumps({{'result_panel_id': p3.name}}))
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
print(json.dumps({{'result_panel_id': p3.name}}))
"""
    return _log_and_execute('Panel_digitize', code)

@mcp.tool()
def Panel_characteristics_snapshots(panel_id: str, month: int | list[int] | None = None) -> str:
    """Create a panel of characteristic snapshots for specific calendar months.

    Args:
        panel_id (str): Identifier for the source characteristic panel to sample.
        month (int | list[int] | None, optional): Single month number or list of month numbers (1-12)
            to capture snapshots for. When ``None`` (default), snapshots are generated for every month.

    Returns:
        str: JSON string containing the persisted panel identifier and metadata of the snapshot results.
    """

    # Serialize the month selector so the sandbox can faithfully reconstruct it.
    #month_payload = json.dumps([] if month is None else month)

    code = f"""
import json
from qrafti import Panel, characteristics_snapshots
characteristics = panel_or_numeric('{panel_id}', **{dates_})
snapshots = characteristics_snapshots(characteristics, month=month).persist()
print(json.dumps({{'result_panel_id': snapshots.name, 'metadata': snapshots.info}}))
"""

    return _log_and_execute('Panel_characteristics_snapshots', code)


@mcp.tool()
def Panel_characteristics_fill(panel_ids: list[str], replace: list | int | float | str | None = None) -> str:
    """Sequentially fill a base characteristic panel using fallback panels.

    Args:
        panel_ids (list[str]): Ordered identifiers of panels whose values should be combined. The
            first panel serves as the base and later panels provide replacement data.
        replace (list | int | float | str | None, optional): Values that should be treated as missing in
            the base panel prior to filling. Scalars are promoted to single-item lists. Defaults to ``None``
            which results in only NaN replacement.

    Returns:
        str: JSON string containing the persisted panel identifier for the filled panel.
    """
    code = f"""
import json
from qrafti import Panel, characteristics_fill
panels = [panel_or_numeric(pid, **{dates_}) for pid in panel_ids]
filled = characteristics_fill(*panels, replace=replace_values).persist()
print(json.dumps({{'result_panel_id': filled.name}}))
"""
    return _log_and_execute('Panel_characteristics_fill', code)


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
from qrafti import Panel, portfolio_impute

weights = panel_or_numeric('{weights_panel_id}', **{dates_})
turnover = portfolio_impute(weights, drifted=True).persist()
print(json.dumps({{'result_panel_id': turnover.name}}))
"""

    return _log_and_execute('Panel_portfolio_turnover', code)


@mcp.tool()
def Panel_sequence(panel_id: str) -> str:
    """Generate a sequential index for each stock across time using cumulative counts.

    Args:
        panel_id (str): Identifier for the panel whose chronological sequence per stock is desired.

    Returns:
        str: JSON string containing the persisted panel identifier and metadata with sequence numbers.
    """
    code = f"""
import json
from qrafti import Panel, cumcount

panel = Panel('{panel_id}', **{dates_})
sequence = panel.trend(cumcount).persist()
print(json.dumps({{'result_panel_id': sequence.name, 'metadata': sequence.info}}))
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
print(json.dumps({{'result_panel_id': p2.name}}))
"""
    return _log_and_execute('Panel_portfolio_weights', code)

@mcp.tool()
def Panel_portfolio_returns(port_weights_panel_id: str) -> str:
    """Compute the portfolio or factor returns given portfolio weights.
    Args:
        port_weights_panel_id (str): The id of the panel data set for portfolio weights.
    Returns:
        str: the id of the created Panel of returns of the factor or portfolio in the cache in JSON format
    """
    code = f"""
import json
from qrafti import Panel, portfolio_returns, panel_or_numeric
p_weights = panel_or_numeric('{port_weights_panel_id}', **{dates_})
p_returns = portfolio_returns(p_weights).persist()
print(json.dumps({{'result_panel_id': p_returns.name}}))
"""
    return _log_and_execute('Panel_portfolio_returns', code)


if __name__ == "__main__":
    # Run with SSE transport
    mcp.run(transport="streamable-http")
