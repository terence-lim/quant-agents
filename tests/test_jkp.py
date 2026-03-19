# python -m tests.test_jkp > tests/jkp.sh
from qrafti import DATES, Panel
from research_utils import rolling, digitize, winsorize, portfolio_weights, portfolio_returns
import pandas as pd
from pathlib import Path

code_prompt = """
You may only use the coding_agent_tool and the Panel_lookup tools;
do not use any other tools to respond to the following query:

"""

q1 = """Define price momentum characteristic as stocks' past 12 months returns skipping one month.
"""

q1a = code_prompt + q1

q2 = q1 + """Sort stocks into characteristic terciles (top/middle/bottom third) with breakpoints 
based on non-micro stocks, where micro stocks are all stocks whose market capitalization 
is below the NYSE 20th percentile.
"""

q2a = code_prompt + q2

q3 = q2 + """For each tercile, compute its "capped value" weighted portfolio, meaning that we weight stocks 
by their market equity winsorized at the NYSE 80th percentile.

The factor returns are then defined as the top-tercile portfolio return minus 
the bottom-tercile portfolio return.
"""

q3a = """
Use a Reflexion-style planning workflow for the query below.

Phase 1 — Initial plan
Read the entire query carefully and draft a step-by-step plan for completing it. 
The plan should be explicit, ordered, and scoped to the full task.
Use your given tools, except the coding_agent tool, and panel data sets.

Phase 2 — Reflection and self-critique
Review the draft plan and evaluate it for completeness, feasibility, and efficiency. Check whether each step:
- can be carried out with the available tools, information, and constraints,
- meaningfully contributes to the objective,
- appears in the right order, and
- avoids unnecessary work or hidden assumptions.

Phase 3 — Revised plan
Revise the plan to address the issues found during reflection, and execute it.

Query:
""" + q3


query_end = """
Return only the panel_id of the final constructed panel, with no additional text. 
If the request cannot be completed because of an error, return exactly `MODEL ERROR`.
"""

out_panels = ['price_momentum', 'code_momentum', 'mom_terciles', 'code_terciles',
              'mom_returns', 'mom_reflexion']
panels = iter(out_panels)

TESTS = Path('tests')
for panel, prompt in zip(out_panels, [q1, q1a, q2, q2a, q3, q3a]):
    query_name = 'test_' + panel
    with open(TESTS / (query_name + ".query"), "w", encoding="utf-8") as f:
        f.write(prompt)
        f.write('\n' + query_end)
    print("python agent_cli.py " + query_name)
    print("python evaluate_agent.py " + query_name)    

#raise Exception

# Compute 12-month-skip-1 momentum    
dates = DATES
window, skip = 12, 1
log1p_ret = Panel().load('RET', **dates).log1p()
factor_pf = log1p_ret.trend(rolling, window=window, skip=skip, agg="sum", interval=1).expm1()
factor_pf.save(next(panels) + '_')
factor_pf.save(next(panels) + '_')

# Compute terciles
nyse_pf = Panel().load("EXCHCD", **dates).apply(pd.DataFrame.isin, values=[1])
size_pf = Panel().load("CAP", **dates)
decile_pf = size_pf.apply(digitize, nyse_pf, cuts=10)
quintile_pf = size_pf.apply(digitize, nyse_pf, cuts=5)
terciles_pf = factor_pf.apply(digitize, reference=decile_pf > 2, cuts=3)
terciles_pf.save(next(panels) + '_')
terciles_pf.save(next(panels) + '_')

# Compute long-short spread returns
vwcap_pf = size_pf.apply(winsorize, nyse_pf, lower=0, upper=0.80)
long_pf = vwcap_pf.apply(portfolio_weights, reference=terciles_pf == 3)
short_pf = vwcap_pf.apply(portfolio_weights, reference=terciles_pf == 1)
long_ret = portfolio_returns(long_pf)
short_ret = portfolio_returns(short_pf)
spreads_pf = long_pf - short_pf
composite_returns = long_ret - short_ret
composite_returns.save(next(panels) + '_')
composite_returns.save(next(panels) + '_')
