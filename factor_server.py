# python server.py
import json
import pandas as pd
from mcp.server.fastmcp import FastMCP
from qrafti import run_code_in_subprocess

dates = dict(start_date='2020-01-01', end_date='2024-12-31')

def log_message(message: str, mode: str = "a"):
    """Log a message to the console and a log file."""
    with open("mcp_server.log", "a") as f:
        f.write(message + "\n")
    print(message)
# log_message(f"MCP server started on {str(datetime.now())}", mode="w")

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

# Create an MCP server

port = 8000
mcp = FastMCP("factor-server", host="0.0.0.0", port=port)

# @mcp.tool()
# def panelframe_weighted_average(panel_id: str, weights_panel_id: str = '') -> str:
#     """Compute weighted average by date.
#     Args:
#         panel_id (str): The id of the panel data set to compute weighted average for.
#         weights_panel_id (str, optional): The id of the weights panel data set to use for weighting the average.
#             If not provided, equal weighting will be used.
#     Returns:
#         str: the id of the created PanelFrame in the cache in JSON format
#     """
#     code = f"""
# import json
# from qrafti import PanelFrame, weighted_average
# p1 = PanelFrame('{panel_id}', **{dates})
# p2 = PanelFrame('{weights_panel_id}', **{dates}) if '{weights_panel_id}' else None
# p3 = p1.apply(weighted_average, None if p2 is None else p2).persist()
# print(json.dumps({{'result_panel_id': p3.name, 'metadata': p3.info}}))
# """
#     log_message(f"\\nExecuting code for panelframe_weighted_average:\\n{code}\\n")
#     return execute_in_sandbox(code)

#
# Specialized Functions 
#
@mcp.tool()
def panelframe_isin(panel_id: str, values: list) -> str:
    """
    Create a PanelFrame that filters the rows of the given panel data 
    to indicate those with values in the provided list.
    Args:
        panel_id (str): The id of the panel data to filter.
        values (list): A list of values to filter the panel data by.
    Returns:
        str: the id of the created PanelFrame in the cache in JSON format
    """
    code = f"""
import json
from qrafti import PanelFrame
import pandas as pd
p1 = PanelFrame('{panel_id}', **{dates})
p2 = p1.apply(pd.DataFrame.isin, values={values}).persist()
print(json.dumps({{'result_panel_id': p2.name, 'metadata': p2.info}}))
"""
    log_message(f"\nExecuting code for panelframe_isin:\n{code}\n")
    return execute_in_sandbox(code)  
    
@mcp.tool()
def panelframe_winsorize(panel_id: str, indicator_panel_id: str = '', lower: float = 0.0, upper: float = 1.0) -> str:
    """
    Create a PanelFrame that winsorizes the values of the given panel data.
    Args:
        panel_id (str): The id of the panel data set to winsorize.
        upper (float): The upper percentile to winsorize to (between 0 and 1).
        lower (float): The lower percentile to winsorize to (between 0 and 1).
        reference_panel_id (str, optional): The id of the indicator panel data set which indicates which
            rows of the panel data set to use for computing the upper and lower bounds.
            If not provided, the winsorization bounds will be computed based on all the values in the panel_id data set.
    Returns:
        str: the id of the created PanelFrame in the cache in JSON format
    """
    code = f"""
import json
from qrafti import PanelFrame, winsorize
p1 = PanelFrame('{panel_id}', **{dates})
p2 = PanelFrame('{indicator_panel_id}', **{dates}) if '{indicator_panel_id}' else None
p3 = p1.apply(winsorize, None if p2 is None else p2, lower={lower}, upper={upper}).persist()
print(json.dumps({{'result_panel_id': p3.name, 'metadata': p3.info}}))
"""
    log_message(f"\\nExecuting code for panelframe_winsorize:\\n{code}\\n")
    return execute_in_sandbox(code)


@mcp.tool()
def panelframe_quantiles(panel_id: str, num: int, indicator_panel_id: str = '') -> str:
    """
    Create a PanelFrame that computes the quantiles of the given panel data.
    Args:
        panel_id (str): The id of the panel data set to compute quantiles for.
        num (int): The number of quantiles to compute.
        reference_panel_id (str, optional): The id of the indicator panel data set which indicates which
            rows of the panel data set to use for computing the quantile breakpoints.
            If not provided, the quantiles will be computed based on all the values in the panel_id data set.
    Returns:
        str: the id of the created PanelFrame in the cache in json format
    """
    code = f"""
import json
from qrafti import PanelFrame, quantiles
p1 = PanelFrame('{panel_id}', **{dates})
p2 = PanelFrame('{indicator_panel_id}', **{dates}) if '{indicator_panel_id}' else None
p3 = p1.apply(quantiles, None if p2 is None else p2, num={num}).persist()
print(json.dumps({{'result_panel_id': p3.name, 'metadata': p3.info}}))
"""
    log_message(f"\nExecuting code for panelframe_quantiles:\n{code}\n")
    return execute_in_sandbox(code)  


@mcp.tool()
def panelframe_spread_portfolios(panel_id: str, weights_panel_id: str = '') -> str:
    """
    Construct long-short spread portfolios of highest and lowest quantiles by date
    Args:
        panel_id (str): The id of the panel data set to construct spread portfolios for.
        weights_panel_id (str, optional): The id of the weights panel data set to use for weighting the portfolios.
            If not provided, equal weighting will be used.
    Returns:
        str: the id of the created PanelFrame in JSON format
    """
    code = f"""
import json
from qrafti import PanelFrame, spread_portfolios
p1 = PanelFrame('{panel_id}', **{dates})
p2 = PanelFrame('{weights_panel_id}', **{dates}) if '{weights_panel_id}' else None
p3 = p1.apply(spread_portfolios, None if p2 is None else p2).persist()
print(json.dumps({{'result_panel_id': p3.name, 'metadata': p3.info}}))
"""
    log_message(f"\\nExecuting code for panelframe_spread_portfolios:\\n{code}\\n")
    return execute_in_sandbox(code)


if __name__ == "__main__":
    # Run with SSE transport
    mcp.run(transport="streamable-http")
