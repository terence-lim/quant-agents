# Use 'factor_agent_tool' for tasks involving characteristic preparation, factor construction, quantile sorting,
# portfolio weighting, and any operations available from the Factor Portfolio Construction Agent such as
# Panel_isin, Panel_winsorize, Panel_quantiles, Panel_spread_portfolios, and
# get_variables_descriptions.
# Use 'risk_agent_tool' for tasks involving generating portfolio returns and performance evaluation
# plotting with Panel_plot, and access to get_variables_descriptions.

# Factor Agent tool catalog:
# - Panel_isin: filter a Panel to identifiers contained in a supplied list.
# - Panel_winsorize: winsorize values using optional indicator Panel and percentile bounds.
# - Panel_quantiles: assign quantile buckets using optional indicator Panel.
# - Panel_spread_portfolios: build long-short spread portfolios with optional weights Panel.
# - get_variables_descriptions: inspect available variables and cache identifiers via the metadata server.

# Risk Agent tool catalog:
# - Panel_matmul: compute matrix multiplication between two Panels.
# - Panel_shift_dates: shift a Panel's dates forward or backward by an integer step.
# - Panel_performance_evaluation: summarize factor performance statistics for a Panel of returns.
# - Panel_plot: create plots for one or two Panels and return the saved image path.
# - get_variables_descriptions: inspect available variables and cache identifiers via the metadata server.


#You may call get_variables_descriptions when you need to understand available variables, but do not delegate
# tasks yourself.
# Describe the computation field with enough detail for the executing agent to know which tool call and
# parameters are needed.

# {panel_id=}, {start_date=}, {end_date=}, {min_stocks=}, {min_value=}, {max_value=}, {dropna=}, 
#        start_date: Keep rows with dates on or after this value, in the form 'YYYY-MM-DD'
#        end_date: Keep rows with dates on or before this value, in the form 'YYYY-MM-DD'
#        min_stocks: Minimum number of stocks per date required to keep that date.
#        min_value: Minimum data value to retain.
#        max_value: Maximum data value to retain.
#        isin: Explicit list of acceptable data values.
#        dropna: If True, drop rows whose values are NaN before persisting.

# python performance_server.py
from mcp.server.fastmcp import FastMCP
from utils import _log_and_execute, dates_, log_message
from pydantic_ai import Agent   # use Agent within a tool call
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import json
import asyncio
import logging
from pydantic_ai.exceptions import UnexpectedModelBehavior
    
from dotenv import load_dotenv
load_dotenv()

# Create an MCP server

port = 8001
mcp = FastMCP("performance-server", host="0.0.0.0", port=port)

#model = "gemini-2.5-flash"
#model = OpenAIChatModel(model_name="gpt-4.1-mini")
#server_agent = Agent(model=model,
#                     system_prompt='You are a helpful assistant specialized in quantitative finance and portfolio performance evaluation.')

async def run_agent_with_timeout(agent, prompt: str, timeout_s: int = 120) -> str:
    """
    Run the pydantic-ai agent with a timeout + resilient error handling.
    Distinguish between client cancellations and model glitches.
    """
    try:
        # Give OpenAI enough time; keep this larger than your MCP tool timeout.
        res = await asyncio.wait_for(agent.run(prompt=prompt), timeout=timeout_s)
        return res.output if hasattr(res, "output") else str(res)

    except asyncio.CancelledError:
        # The MCP client or framework cancelled the tool call (rerun/timeout/disconnect).
        logging.warning("Agent run was cancelled by the caller (tool timeout/rerun/disconnect).")
        # Re-raise if you want the tool to hard-cancel, OR return a friendly string:
        return "Request was cancelled by the caller (timeout or rerun). Please try again."

    except UnexpectedModelBehavior as e:
        # Gemini/OpenAI occasionally return empty payloads; surface a readable message.
        logging.warning("Model returned unexpected payload: %s", e)
        return ("Agent could not complete the step because the model returned an empty payload. "
                "Please try again or switch models.")

    except Exception as e:
        logging.exception("Agent run failed:")
        return f"Agent error: {e}"


@mcp.tool(timeout=180)
async def Panel_factor_evaluate(factor_panel_id: str, description: str = '') -> str:
    """Provides a prompt with instructions and context to generate a research report 
    for evaluating a Panel of factor characteristic values for predicting stock returns.

    Args:
        factor_panel_id (str): The id of the panel data set for factor characteristic values.
        description (str): A full description and definition of the factor.
    Returns:
        str: A prompt containing instructions and context for generating a research report evaluating the factor.
    """
    # TO DO: 
    # (1) save factor_evaluation to a temp folder 
    # (2) return path as json 
    # (3) include prompt to save report after generation
    description = description.replace('"', "'")  # sanitize quotes
    code = f"""
import json
from qrafti import Panel, factor_evaluate, MEDIA, research_prompt
panel = Panel('{factor_panel_id}', **{dates_})
evaluate_str = factor_evaluate(panel)
#subfolder = 'folder_0'
#output_folder = MEDIA / subfolder
#output_folder.mkdir(parents=True, exist_ok=True)
#with open(output_folder / 'research_memo.md', 'w') as f:
#    f.write(evaluate_str)
#prompt_string = "{description}" + "\\n---------\\n" + research_prompt + "\\n\\n" + evaluate_str
#print(json.dumps(dict(prompt_string=prompt_string, output_folder=subfolder)))
print(evaluate_str)
"""
    # payload = _log_and_execute('Panel_factor_evaluate', code)
    # if 'prompt_string' in payload:
    #     result = json.loads(payload)
    #     prompt_string = result['prompt_string']
    #     log_message('Server Calling Agent With Prompt', 'Panel_factor_evaluate', prompt_string)
    #     #response = await run_agent_with_timeout(server_agent, prompt_string, timeout_s=180)
    #     #response = await server_agent.run(prompt=prompt_string)
    #     response = await asyncio.wait_for(server_agent.run(prompt=prompt), timeout=timeout_s)
    #     response = response.output if hasattr(response, "output") else str(response)
    #     log_message('Server Agent Returned Payload', 'Panel_factor_evaluate', response)
    #     return f"Research Memo:\n\n{response}"
    # else:
    #     return payload        

SPECIALIZED_AGENT_TOOLS: Dict[str, List[Dict[str, str]]] = {
    "factor_agent_tool": [
        {
            "name": "Panel_characteristics_downsample",
            "description": "Down-samples or filters characteristic by selected calendar months.",
        },
        {
            "name": "Panel_characteristics_fill",
            "description": "If values are not available, then sequentially fill from list of panels in order.",
        },
        {
            "name": "Panel_sequence",
            "description": "Compute the cumulative number of available data points for each stock observation",
        },
        {
            "name": "Panel_winsorize",
            "description": "Winsorize panel values using optional reference weights and percentile cutoffs.",
        },
        {
            "name": "Panel_digitize",
            "description": "Discretize panel observations into categories with optional masking and ordering controls.",
        },
        {
            "name": "Panel_portfolio_turnover",
            "description": "Measure turnover by imputing drifted portfolio weights with optional return inputs.",
        },
        {
            "name": "Panel__portfolio_weights",
            "description": "Construct portfolio weights from raw scores or stock characteristics with optional masking and leverage scaling.",
        },
    ],
    "performance_agent_tool": [
        {
            "name": "Panel_performance_evaluation",
            "description": "Summarize risk and performance statistics for a returns panel.",
        },
        {
            "name": "Panel_portfolio_returns",
            "description": "Generate portfolio or factor returns from portfolio weights or factor characteristic panels.",
        },
        {
            "name": "Panel_plot",
            "description": "Render and persist plots for one or two panels with configurable plot type and title.",
        },
    ],
}

def _summarize_tools(functions: List[Callable[..., str]]) -> List[Dict[str, str]]:
    """Return name/description metadata for the provided MCP tool callables."""

    summaries: List[Dict[str, str]] = []
    for func in functions:
        doc = (func.__doc__ or "").strip()
        description = doc.splitlines()[0] if doc else ""
        summaries.append({"name": func.__name__, "description": description})
    return summaries


def _server_tool_metadata() -> List[Dict[str, str]]:
    """Collect metadata for MCP tools defined in this server module."""

    tool_functions: List[Callable[..., str]] = [
        # Panel_matmul,
        Panel_add,
        Panel_radd,
        Panel_sub,
        Panel_rsub,
        Panel_mul,
        Panel_rmul,
        Panel_truediv,
        Panel_rtruediv,
        Panel_eq,
        Panel_ne,
        Panel_lt,
        Panel_le,
        Panel_gt,
        Panel_ge,
        Panel_or,
        Panel_and,
        Panel_neg,
        Panel_invert,
        Panel_filter,
        Panel_log,
        Panel_exp,
        Panel_shift
    ]
    return _summarize_tools(tool_functions)


@mcp.tool()
def get_specialized_agent_tools() -> str:
    """Return specialized and shared MCP tools accessible to planner workflows."""

    payload = {
        "agents": SPECIALIZED_AGENT_TOOLS,
        "server": _server_tool_metadata(),
        "notes": "Each entry lists the tools callable by a specialized agent tool plus shared server MCP tools they may invoke.",
    }
    return json.dumps(payload)




import pandas as pd
from qrafti import Panel, Calendar, DATE_NAME, STOCK_NAME
def factor_generate(factor: Panel, lags: int, window: int, univ: Panel = None) -> Panel:
    """Generate a factor Panel from rolling windows based on universe filter.
    Arguments:
        lags: Number of months to lag the factor values
        window: Window size for rolling accumulation of factor values
        univ: Optional Panel of universe filter
    Returns:
        Panel of generated factor values
    """
    assert factor.nlevels == 2, "Factor must have two index levels"
    cal = Calendar()
    factor_dates = factor.dates
    start_date = cal.offset(factor_dates[0], offset=lags, strict=False)
    end_date = cal.offset(factor_dates[-1], offset=lags, strict=False)
    factor_final = []
    for next_date in cal.dates_range(start_date, end_date):
        # For each date, collect data from lagged window
        start_window = cal.offset(next_date, offset = -(window + lags), strict=False)
        end_window = cal.offset(next_date, offset = -lags, strict=False)
        for curr_date in cal.dates_range(start_window, end_window):
            if curr_date in factor_dates:
                factor_df = factor.frame.xs(curr_date, level=0).reset_index()
                factor_df[DATE_NAME] = next_date
                factor_df['_date_'] = curr_date
                factor_final.append(factor_df)

    # sort by STOCK_NAME, DATE_NAME and _date_ and drop duplicates, keep last
    factor_final = pd.concat(factor_final, axis=0)
    factor_final = factor_final.sort_values(by=[STOCK_NAME, DATE_NAME, '_date_'])
    factor_final = factor_final.drop_duplicates(subset=[STOCK_NAME, DATE_NAME], keep='last')
    factor_final = factor_final.set_index([DATE_NAME, STOCK_NAME]).drop(columns=['_date_'])

    # require index to be in univ.frame
    factor_final = factor_final.join(univ.frame, how='inner', rsuffix='_univ').iloc[:, :1]
    factor_final = Panel().set_frame(factor_final)
    return factor_final


def weighted_average(x):
    """
    Compute the weighted average of the first column, weighted by the last column.
    Arguments:
        x: DataFrame with at least two columns, first column is the data to be averaged,
           last column is the weight for each row
    Returns:
        float: Weighted average of the first column
    Usage:
        panel_frame.apply(weighted_average, weights or 1, fill_value=0)
    """
    return (x.iloc[:, 0] * x.iloc[:, 1]).sum() / x.iloc[:, 1].sum()

# @mcp.tool()
# def panel_weighted_average(panel_id: str | int | float, weights_panel_id: str | int | float = '') -> str:
#     """Compute weighted average by date.
#     Args:
#         panel_id (str): The id of the panel data set to compute weighted average for.
#         weights_panel_id (str, optional): The id of the weights panel data set to use for weighting the average.
#             If not provided, equal weighting will be used.
#     Returns:
#         str: the id of the created panel in the cache in JSON format
#     """
#     code = f"""
# import json
# from qrafti import Panel, weighted_average, panel_or_numeric
# p1 = panel_or_numeric('{panel_id}'), **{dates_})
# p2 = panel_or_numeric('{weights_panel_id}'), **{dates_})
# p3 = p1.apply(weighted_average, None if p2 is None else p2).persist()
# print(json.dumps({{'result_panel_id': p3.name, 'metadata': p3.info}}))
# """
#     log_message(f"\\nExecuting code for Panel_weighted_average:\\n{code}\\n")
#     return execute_in_sandbox(code)

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


#
# Specialized Functions 
#
# {
#     "name": "Panel_spread_portfolios",
#     "description": "Construct long-short spread portfolio weights between highest and lowest quantiles, using optional weights.",
# },


import matplotlib.pyplot as plt
import seaborn as sns

data = [44, 45, 40, 41, 39]
labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']

# declaring exploding pie
explode = [0, 0.1, 0, 0, 0]

# define Seaborn color palette to use
colors = sns.color_palette('dark')

# plotting data on chart
plt.pie(data, labels=labels, colors=colors, explode=explode, autopct='%.0f%%')
plt.show()
"""

country_dict = {
    'USA': "United States",
    "JPN": "Japan",
    "GBR": "United Kingdom",
    "CAN": "Canada",
    "FRA": "France",
    "DEU": "Germany",
    "AUS": "Australia",
    "CHE": "Switzerland",
    "NLD": "Netherlands",
    "HKG": "Hong Kong",
    "ESP": "Spain",
    "SWE": "Sweden",
    "DNK": "Denmark",
    "ITA": "Italy",
    "BEL": "Belgium",
    "SGP": "Singapore",
    "NOR": "Norway",
    "FIN": "Finland",
    "ISR": "Israel",
    "AUT": "Austria",
    "IRL": "Ireland",
    "NZL": "New Zealand",
    "PRT": "Portugal",
}

"""

'''
class PortfolioEvaluation:
    """Evaluate the performance of a portfolio DataFrame
        if portfolio.nlevels != 2:
            raise ValueError("PortfolioEvaluation requires a Panel with 2 index levels (date, stock)")
        PortfolioEvaluation(portfolio.frame)
    """

    def __init__(self, portfolio: pd.DataFrame):
        if portfolio.nlevels != 2:
            raise ValueError("PortfolioEvaluation requires a Panel with 2 index levels (date, stock)")
        self.portfolio = portfolio

    def turnover(self, ret: Panel) -> Panel:
        """Compute the turnover of the portfolio as the sum of absolute changes in weights.
        Arguments:
            ret: Panel of leading returns to compute drifted portfolio weights
        Returns:
            Panel of turnover values for each date
        """
        # shift both portfolio and returns by 1 period to align
        ret = ret.shift_dates(shift=1)
        shifted_portfolio = self.portfolio.shift_dates(shift=1)

        # left join shifted portfolio with 1 + returns, and multiply to get drifted weights
        df = shifted_portfolio.join_frame(ret + 1, fillna=1, how='left')
        df.iloc[:, 0] = df.iloc[:, 0] * df.iloc[:, 1]  # drift weights by returns
        shifted_portfolio.set_frame(df.iloc[:, [0]])  # update shifted portfolio weights

        # join original portfolio with drifted portfolio weights
        df = self.portfolio.join_frame(shifted_portfolio, fillna=0, how='left')

        # compute turnover as sum of absolute changes in weights
        turnover = df.groupby(level=0).apply(lambda x: (x.iloc[:, 0] - x.iloc[:, -1]).abs().sum())

        return Panel().set_frame(turnover)
    
    
    def information_coefficient(self, ret: Panel) -> Panel:
        """Compute the Information Coefficient (IC) of the factor against the given returns.
        Arguments:
            ret: Panel of returns to compute IC against
        Returns:
            Panel of IC values for each date
        """
        def ic_func(x):
            return x.iloc[:, 0].corr(x.iloc[:, 1])
        return self.portfolio.apply(ic_func, ret, fillna=0)
'''

TODO:
- def TimeFrame.performance(): annualized return, volatility, sharpe ratio, max drawdown
- def Panel.turnover()
- Download and check JKP factors
"""

import matplotlib.pyplot as plt
import seaborn as sns

def pie_chart(data, labels, title: str, ncol=3):
    """Plot a pie chart with the given data and labels."""
    colors = sns.color_palette('pastel')

    fig, ax = plt.subplots(figsize=(8, 6))

    wedge_width = 0.5
    pctdistance = 1 - wedge_width / 2

    wedges, texts, autotexts = ax.pie(
        data,
        labels=labels,                 # show labels on wedges
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=wedge_width),
        autopct=lambda pct: f"{pct:.0f}%",
        pctdistance=pctdistance
    )

    # --- Adjust label styles ---
    for text in texts:
        text.set_fontsize(16)          # category labels (outside wedges)
#        text.set_weight('bold')
        text.set_color('black')

    for autotext in autotexts:
        autotext.set_fontsize(14)        # increase font size
        autotext.set_color('black')
#        autotext.set_weight('bold')

    # bold title, large font
    ax.set_title(title, fontweight='bold', fontsize=18)

    # Adjust legend placement: lower center, wide layout
    ax.legend(
        wedges,
        labels,
        fontsize=14,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.1),  # center below chart
        ncol=ncol,
        #ncol=(len(labels)+1)//2,            # all items in one row
        frameon=False
    )

if __name__ == "__main__":
    import re
    from pathlib import Path
    filename = Path('/home/terence/Downloads/scratch/2024/JKP/variables.txt.orig')
    names, types, descriptions = [], [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().lower().startswith('name '):
                continue  # skip header
            # Use regex to extract: name type length description
            match = re.match(r'^(\S+)\s+(\S+)\s+\S+\s+(.*)', line.strip())
            if match:
                name, typ, desc = match.group(1), match.group(2), match.group(3)
                names.append(name)
                types.append(typ)
                descriptions.append(desc)
    df = pd.DataFrame({'Type': types, 'Description': descriptions}, index=names)
    df.index.name = 'Name'

if False:
    data = {"GPT family":58, "Claude":13, "LLaMA":11, "Gemini": 11, "other":7}
    pie_chart(data.values(), data.keys(), title="Models Employed (%)")
    plt.tight_layout()
    plt.savefig("models.svg", bbox_inches='tight')

    data = {"Social media": 25, "Education": 21, "Software": 20, "Healthcare": 16, "Arts/Humanities": 18}
    pie_chart(data.values(), data.keys(), title="Datasets by Domain (%)")
    plt.tight_layout()
    plt.savefig("domain.svg", bbox_inches='tight')

    data = {"Zero-shot":35, "Few-shot":16, "Chain-of-thought": 13, "Reflexion":15, "Tool/agent":11}
    pie_chart(data.values(), data.keys(), title="Prompting Strategies (%)")
    plt.tight_layout()
    plt.savefig("prompting.svg", bbox_inches='tight')

    data = {"Human review":40, "Similarity metrics":27, "Task-based":13, "hybrid":20}
    pie_chart(data.values(), data.keys(), title="Evaluation Fragmentation (%)")
    plt.tight_layout()
    plt.savefig("evaluation.svg", bbox_inches='tight')

    data = dict(Inductive=64, Hybrid=22, Deductive=9)
    pie_chart(data.values(), data.keys(), title="Type of Thematic Analysis (%)")
    plt.tight_layout()
    plt.savefig("TA.svg", bbox_inches='tight')
    
    plt.show()


