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


# @mcp.tool()
# def Panel_spread_portfolios(panel_id: str, weights_panel_id: str | int | float = '') -> str:
#     """
#     Construct spread portfolio weights as long the highest and short the lowest quantile stocks
#     Args:
#         panel_id (str): The id of the quantiles panel data set to construct spread portfolio weights for.
#         weights_panel_id (str, optional): The id of the panel data set to use for weighting stocks in portfolios.
#             If not provided, equal weighting will be used.
#     Returns:
#         str: the id of the created Panel of stock weights in the spread portfolio in JSON format
#     """
#     code = f"""
# import json
# from qrafti import Panel, spread_portfolios, panel_or_numeric
# p1 = panel_or_numeric('{panel_id}', **{dates_})
# p2 = panel_or_numeric('{weights_panel_id}', **{dates_})
# p3 = p1.apply(spread_portfolios, None if p2 is None else p2).persist()
# print(json.dumps({{'result_panel_id': p3.name, 'metadata': p3.info}}))
# """
#     log_message(f"\\nExecuting code for Panel_spread_portfolios:\\n{code}\\n")
#     return execute_in_sandbox(code)


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
