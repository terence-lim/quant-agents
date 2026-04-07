# python -m tests.test_ff > tests/ff.sh
from qrafti import DATES, Panel
from research_utils import (characteristics_coalesce, characteristics_resample, digitize,
                            portfolio_weights, portfolio_returns)
from pathlib import Path

code_prompt = """
You may only use the coding_agent_tool and the Panel_lookup tools;
do not use any other tools to respond to the following query:

"""

q1 = """Construct book equity as stockholder equity (seq) minus the book value of preferred stock, 
where preferred stock is defined as the redemption (pstkrv), liquidation (pstkl), 
or par value (pstk), if available, in that order.
"""

q1a = code_prompt + q1

q2 = q1 + """Construct book equity as stockholder equity (seq) minus the book value of preferred stock, 
where preferred stock is defined as the redemption (pstkrv), liquidation (pstkl), 
or par value (pstk), if available, in that order.

Each December, divide book equity of a firm’s fiscal year-end at or before December 
by the company market capitalization (CAPCO) at the end of the year. 
Then construct book-to-market equity in June as the lagged values from the previous December.
"""

q2a = code_prompt + q2

q3 = q2 + """Sort stocks independently on market equity (CAP) into small and big stocks using 
the median market capitalization of all stocks traded on the NYSE as breakpoints, 
and on book-to-market equity into growth, neutral, and value stocks using the 30th and 70th 
percentiles of book-to-market equity of all stocks traded on the NYSE as breakpoints.

Form portfolios as the intersection of these sorts, 
where the portfolios are weighted by market equity (CAP).

Construct the HML factor portfolio as the average of a small and a big value portfolio minus 
the average of a small and a big growth portfolio in each month. 
"""

q3a = code_prompt + q3

'''
q4a = """
Use a Reflexion-style planning workflow for the query below.

Phase 1 — Initial plan
Read the entire query carefully and draft a step-by-step plan for completing it. The plan should be explicit, ordered, and scoped to the full task.

Phase 2 — Reflection and self-critique
Review the draft plan and evaluate it for completeness, feasibility, and efficiency. Check whether each step:
- can be carried out with the available tools, information, and constraints,
- meaningfully contributes to the objective,
- appears in the right order, and
- avoids unnecessary work or hidden assumptions.

Phase 3 — Revised plan
Revise the plan to address the issues found during reflection, and execute it.

Query:
""" + q4

q4a = """
Use a Reflexion-style planning workflow for the query below.

Phase 1: Suggest a sequential order of steps using your given tools, except coding_agent_tool, and panel data to construct the factor.

Phase 2: Check that each step is implementable with available tools and that the steps can efficiently satisfy the query.

Phase 3: Rewrite the plan, and execute the corrected plan.

Query:
""" + q4
'''

q3a = """
Use a Reflexion-style planning workflow for the query below.

Phase 1 — Initial plan:
Read the entire query carefully and draft a step-by-step plan for completing it. 
Use your given tools, except coding_agent_tool, and panel data sets.

Phase 2 — Reflection and self-critique:
Review the draft plan and evaluate it for completeness, feasibility, and efficiency. 
Check whether each step can be carried out with the available tools and constraints.

Phase 3 — Revised plan:
Revise the plan to address the issues found during reflection, and execute it.

Query:
""" + q3

q3b = code_prompt + q3

query_end = """
Return the id of the final constructed panel.
If the request cannot be completed because of an error, return exactly `MODEL ERROR`.
"""
#Return only the results_panel_id of the final constructed panel, with no additional text.

out_panels = ['book_value', 'code_value', 'book_market', 'code_market',
              'hml_returns', 'hml_reflexion', 'hml_code']
panels = iter(out_panels)

TESTS = Path('tests')
for panel, prompt in zip(out_panels, [q1, q1a, q2, q2a, q3, q3a, q3b]):
    query_name = 'test_' + panel
    with open(TESTS / (query_name + ".query"), "w", encoding="utf-8") as f:
        f.write(prompt)
        f.write('\n' + query_end)
    print("python agent_cli.py " + query_name)
    print("python evaluate_agent.py " + query_name)    


raise Exception

dates = DATES
years = Panel().load('YEARS', **dates)
bench_id = 'HML'
panel_counter = 0

# Compute Book Value
pstkrv = Panel().load("pstkrv", **dates)
pstkl = Panel().load("pstkl", **dates)
pstk = Panel().load("pstk", **dates)
seq = Panel().load("seq", **dates)  # total shareholders' equity
preferred_stock = characteristics_coalesce(pstkrv, pstkl, pstk)
#txditc = Panel().load("txditc", **dates).restrict(end_date="1993-12-31")
book_value = (seq - preferred_stock).save(next(panels) + '_')
book_value.save(next(panels) + '_')

# Compute Book to Market at December samples
month = 12  # Decembers
book_samples = characteristics_resample(book_value, ffill=True, month=[month])
company_value = Panel().load("CAPCO", **dates)
company_value = characteristics_resample(company_value, ffill=True, month=[month])
    
# Lag Book to Market, restrict to universe and form terciles based on NYSE stocks
lags = 6
book_market = (book_samples / company_value).shift(lags).save(next(panels) + '_')
book_market.save(next(panels) + '_')


"""
# Form size and bm quantiles based on NYSE stocks
nyse = Panel().load("EXCHCD", **dates) == 1
book_quantiles = book_market.apply(digitize, reference=nyse, cuts=[0.3, 0.7])
size_quantiles = Panel().load("CAP", **dates)
size_quantiles = size_quantiles.apply(digitize, nyse, cuts=2)

# Form intersection
market_value = Panel().load("CAP", **dates)
BL1 = market_value.apply(portfolio_weights, reference=(size_quantiles == 2) & (book_quantiles == 1))
BH1 = market_value.apply(portfolio_weights, reference=(size_quantiles == 2) & (book_quantiles == 3))
SL1 = market_value.apply(portfolio_weights, reference=(size_quantiles == 1) & (book_quantiles == 1))
SH1 = market_value.apply(portfolio_weights, reference=(size_quantiles == 1) & (book_quantiles == 3))

BL = portfolio_returns(BL1)
BH = portfolio_returns(BH1)
SL = portfolio_returns(SL1)
SH = portfolio_returns(SH1)

composite_returns = ((SH - SL + BH - BL) / 2).save(next(panels) + '_')
composite_returns.save(next(panels) + '_')

# print(out_panels)

"""

hml = Panel().load("HML")
hml.save(next(panels) + '_')
hml.save(next(panels) + '_')
hml.save(next(panels) + '_')

