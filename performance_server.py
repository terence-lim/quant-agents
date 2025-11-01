# python performance_server.py
from mcp.server.fastmcp import FastMCP
from utils import _log_and_execute, dates_
    
# Create an MCP server

port = 8001
mcp = FastMCP("performance-server", host="0.0.0.0", port=port)

@mcp.tool()
def Panel_performance_evaluation(panel_id: str, benchmark_panel_id: str = '') -> str:
    """
    Evaluate the performance of portfolio returns in a Panel
    Args:
        panel_id (str): The id of the panel data set to evaluate.
    Returns:
        str: Performance evaluation metrics in JSON format.
    """
    code = f"""
import json
from qrafti import Panel, portfolio_evaluation, panel_or_numeric
p1 = panel_or_numeric('{panel_id}', **{dates_})
summary_dict = portfolio_evaluation(p1)
print(json.dumps(summary_dict))
"""
    return _log_and_execute('Panel_performance_evaluation', code)

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
    code = f"""
import json
from qrafti import Panel, MEDIA, panel_or_numeric
import matplotlib.pyplot as plt
p1 = panel_or_numeric('{panel_id}', **{dates_})
p2 = panel_or_numeric('{other_panel_id}', **{dates_})
if p2:
    fig = p1.plot(p2, kind='{kind}', title='{title}')
else:
    fig = p1.plot(kind='{kind}', title='{title}')
savefig = MEDIA / f"plot_{panel_id}.png"
plt.savefig(savefig)
print(json.dumps({{'image_path_name': 'file://' + str(savefig)}}))
"""
    return _log_and_execute('Panel_plot', code)

if __name__ == "__main__":
    # Run with SSE transport
    mcp.run(transport="streamable-http")
