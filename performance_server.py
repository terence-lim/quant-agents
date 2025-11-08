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

performance_prompt = f"""You are a sell-side quantitative researcher writing a captivating research memo
on this new financial signal for predicting stock returns. You should also provide a title name for the signal.

Please follow these guidelines for writing the research memo:

1. Motivation (1 paragraph, ~100 words): 
    * Broad statement on market efficiency or asset pricing. 
    * Identify a gap in the current practice and literature.
    * Use active voice and declarative statements.

2. Hypothesis Development (1 paragraph, ~150 words):
    * Present economic mechanisms linking signal to returns.
    * Draw on theoretical frameworks.
    * Support claims with citations.

3. Results Summary (1-2 paragraphs, ~200 words):
    * Lead with the strongest statistical finding.
    * Summarize the key results in a narrative form, including economic significance.
    * Do not merely cite numbers; interpret them.

4. Contribution (1 paragraph, ~150 words):
    * Position relative to 3-4 related finance/accounting journal articles.
    * Highlight methodological innovations.

In your writing, please:

* Use active voice (e.g., "We find").
* Maintain clarity and conciseness.
* Avoid jargon; explain technical terms.
* Use present tense for established findings.
* Use past tense for specific results.
* Make clear distinctions between correlation and causation.
* Avoid speculation beyond the data.

Output in markdown format with sections: Introduction, Hypothesis Development, Results, Contribution.

Base the results section strictly on the following data, matching its terminology and precision:
"""


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

    panel = Panel(factor_panel_id, **DATES)
    evaluate_str = factor_evaluate(panel)
    payload = dict(output_str=evaluate_str, description=description)
    memo_prompt = "\n\n".join([performance_prompt.strip(), evaluate_str.strip()])
    payload = dict(
        output_str=evaluate_str,
        description=description,
        memo_prompt=memo_prompt,
    )
    log_message(str(payload), "Panel_factor_evaluate", f"{factor_panel_id=}, {description=}")
    return json.dumps(payload)


if __name__ == "__main__":
    # Run with SSE transport
    mcp.run(transport="streamable-http")
