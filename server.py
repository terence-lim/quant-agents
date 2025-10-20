# python server.py
import json
import sys
import pandas as pd
from mcp.server.fastmcp import FastMCP
from qrafti import load_variables, run_code_in_subprocess
from dotenv import load_dotenv
load_dotenv()

#
# Global variables -- should be in Memory
# 
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

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
mcp = FastMCP("cache-server", host="0.0.0.0", port=port)

@mcp.tool()
def panelframe_matmul(panel_id: str, other_panel_id: str) -> str:
    """Compute the dot product (matrix multiplication) between two PanelFrames.
    Args:
        panel_id (str): The id of the first panel data set.
        other_panel_id (str): The id of the second panel data set.
    Returns:
        str: the id of the created PanelFrame in the cache in JSON format
    """
    code = f"""
import json
from qrafti import PanelFrame
p1, p2 = PanelFrame('{panel_id}', **{dates}), PanelFrame('{other_panel_id}', **{dates})
p3 = (p1 @ p2).persist()
print(json.dumps({{'result_panel_id': p3.name}}))
"""
    log_message(f"\\nExecuting code for panelframe_matmul:\\n{code}\\n")
    return execute_in_sandbox(code)


@mcp.tool()
def panelframe_weighted_average(panel_id: str, weights_panel_id: str = '') -> str:
    """Compute weighted average by date.
    Args:
        panel_id (str): The id of the panel data set to compute weighted average for.
        weights_panel_id (str, optional): The id of the weights panel data set to use for weighting the average.
            If not provided, equal weighting will be used.
    Returns:
        str: the id of the created PanelFrame in the cache in JSON format
    """
    code = f"""
import json
from qrafti import PanelFrame, weighted_average
p1 = PanelFrame('{panel_id}', **{dates})
p2 = PanelFrame('{weights_panel_id}', **{dates}) if '{weights_panel_id}' else None
p3 = p1.apply(weighted_average, None if p2 is None else p2).persist()
print(json.dumps({{'result_panel_id': p3.name}}))
"""
    log_message(f"\\nExecuting code for panelframe_weighted_average:\\n{code}\\n")
    return execute_in_sandbox(code)

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
print(json.dumps({{'result_panel_id': p2.name}}))
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
print(json.dumps({{'result_panel_id': p3.name}}))
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
print(json.dumps({{'result_panel_id': p3.name}}))
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
print(json.dumps({{'result_panel_id': p3.name}}))
"""
    log_message(f"\\nExecuting code for panelframe_spread_portfolios:\\n{code}\\n")
    return execute_in_sandbox(code)

@mcp.tool()
def panelframe_shift_dates(panel_id: str, shift: int = 1) -> str:
    """
    Create a new PanelFrame with its date index shifted forward, by one date by default.

    This function moves all data forward in time along the date index.
    The dates with no values to shift from in the dataset are dropped.

    Common use cases include aligning lagged or forward-looking features in 
    time-series and panel data analysis.

    Args:
        panel_id (str): 
            The ID of the existing PanelFrame to shift. This should correspond to 
            a PanelFrame object stored in the cache.
        shift (int, optional): 
            The number of date steps to shift forward. Default is 1.

    Returns:
        str: 
            The ID of the newly created PanelFrame (stored in the cache) whose dates 
            have been shifted forward by one step.
    """
    code = f"""
import json
from qrafti import PanelFrame
p1 = PanelFrame('{panel_id}', **{dates})
p2 = p1.shift_dates(shift={shift}).persist()
print(json.dumps({{'result_panel_id': p2.name}}))
"""
    log_message(f"\\nExecuting code for panelframe_dates_shift:\\n{code}\\n")
    return execute_in_sandbox(code)

@mcp.tool()
def panelframe_performance_evaluation(panel_id: str, benchmark_panel_id: str = '') -> str:
    """
    Evaluate the performance of portfolio returns in a PanelFrame
    Args:
        panel_id (str): The id of the panel data set to evaluate.
    Returns:
        str: Performance evaluation metrics in JSON format.
    """
    code = f"""
import json
from qrafti import PanelFrame, FactorEvaluation
p1 = PanelFrame('{panel_id}', **{dates})
summary_dict = FactorEvaluation(p1).summary()
print(json.dumps(summary_dict))
"""
    log_message(f"\\nExecuting code for panelframe_performance_evaluation:\\n{code}\\n")
    return execute_in_sandbox(code)

@mcp.tool()
def panelframe_plot(panel_id: str, other_panel_id: str = '', kind: str ='line', title='') -> str:
    """
    Plot the values of a PanelFrame.
    Optionally, plot it together with another PanelFrame on the same axes.

    Args:
        panel_id (str): The ID of the primary PanelFrame to plot.
        other_panel_id (str, optional): The ID of another PanelFrame to plot together.
        kind: (str, optional): The type of plot to create (e.g., 'line', 'bar', 'scatter').
        title (str, optional): The title of the plot.
    Returns:
        str: Full path name containing the plot image in JSON format.
    """
    code = f"""
import json
from qrafti import PanelFrame, MEDIA_PATH
import matplotlib.pyplot as plt
p1 = PanelFrame('{panel_id}', **{dates})
p2 = PanelFrame('{other_panel_id}', **{dates}) if '{other_panel_id}' else None
if p2:
    fig = p1.plot(p2, kind='{kind}', title='{title}')
else:
    fig = p1.plot(kind='{kind}', title='{title}')
savefig = MEDIA_PATH / f"plot_{panel_id}.png"
plt.savefig(savefig)
print(json.dumps({{'image_path_name': 'file://' + str(savefig)}}))
"""
    log_message(f"\\nExecuting code for panelframe_plot:\\n{code}\\n")
    return execute_in_sandbox(code)

@mcp.tool()
def execute_python(code: str) -> str:
    """
    Safely execute a Python code string and return the output in markdown format.

    Args:
        code_str (str): The Python code to execute.
    Returns:
        str: The standard output from the executed code, preferably in JSON format.
    """
    log_message(f"\nExecuting arbitrary python code:\n{code}\n")
    return execute_in_sandbox(code)


@mcp.tool()
def get_variables_descriptions() -> dict:
    """
    Get the PanelFrame id's and descriptions of all the variables about stocks in the database.
    The variables include stocks' prices, returns, fundamentals, and technical indicators.

    Returns:
        A dictionary with PanelFrame id's as keys and their descriptions as values.
    """
    df = load_variables()
    # Convert Series to dict for compatibility
    log_message("Fetching characteristic descriptions.")
    return df["Description"].to_dict()

if __name__ == "__main__":
    # Run with SSE transport
    mcp.run(transport="streamable-http")
