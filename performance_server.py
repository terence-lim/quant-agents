# python performance_server.py
from mcp.server.fastmcp import FastMCP
from utils import _log_and_execute, log_message
from pydantic_ai import Agent   # use Agent within a tool call
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import json
import logging
import asyncio
from pydantic_ai.exceptions import UnexpectedModelBehavior

import json
from qrafti import Panel, panel_or_numeric, str_or_None, numeric_or_None, int_or_None, DATES
from qrafti import factor_evaluate
from dotenv import load_dotenv
load_dotenv()

# Create an MCP server

port = 8001
mcp = FastMCP("performance-server", host="0.0.0.0", port=port)

@mcp.tool()
def Panel_factor_evaluate(factor_panel_id: str, description: str = '') -> str:
    """Returns summary tables evaluating a Panel of characteristics data for predicting stock returns.

    Args:
        factor_panel_id (str): The id of the panel data set for factor characteristic values.
        description (str): A full description and definition of the factor.
    Returns:
        str: Summary tables in markdown text format
    """
    log_message(tool="Panel_factor_evaluate", code=f"{factor_panel_id=}, {description=}")
    panel = Panel(factor_panel_id, **DATES_)
    evaluate_str = factor_evaluate(panel)
    payload = dict(output_str=evaluate_str, description=description)
    log_message(str(payload), "Panel_factor_evaluate", f"{factor_panel_id=}, {description=}")
    return json.dumps(payload)


if __name__ == "__main__":
    # Run with SSE transport
    mcp.run(transport="streamable-http")
