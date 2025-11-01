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
# @mcp.tool()
# def Panel_isin(panel_id: str | int | float, values: list) -> str:
#     """
#     Create a Panel that filters the rows of the given panel data 
#     to indicate those with values in the provided list.
#     Args:
#         panel_id (str): The id of the panel data to filter.
#         values (list): A list of values to filter the panel data by.
#     Returns:
#         str: the id of the created Panel in the cache in JSON format
#     """
#     code = f"""
# import json
# from qrafti import Panel, panel_or_numeric
# import pandas as pd
# p1 = panel_or_numeric('{panel_id}', **{dates_})
# p2 = p1.apply(pd.DataFrame.isin, values={values}).persist()
# print(json.dumps({{'result_panel_id': p2.name, 'metadata': p2.info}}))
# """
#     log_message(f"\nExecuting code for Panel_isin:\n{code}\n")
#     return execute_in_sandbox(code)  
#        {
#            "name": "Panel_isin",
#            "description": "Create a boolean mask panel indicating rows whose values appear in a provided list.",
#        },
# {
#     "name": "Panel_spread_portfolios",
#     "description": "Construct long-short spread portfolio weights between highest and lowest quantiles, using optional weights.",
# },

"""
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


