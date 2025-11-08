# python coding_server.py
from mcp.server.fastmcp import FastMCP
from qrafti import load_variables
from utils import _log_and_execute, log_message
import json
from typing import Any, List, Optional, Dict, Callable

import json
from qrafti import Panel, panel_or_numeric, str_or_None, numeric_or_None, int_or_None, DATES
from qrafti import run_code_in_subprocess


#import logging
#logging.basicConfig(level=logging.DEBUG)

# Create an MCP server
mcp = FastMCP("coding-server", host="0.0.0.0", port=8003)

@mcp.tool()
def execute_python(code_str: str) -> str:
    """
    Execute a Python code string and returns its standard output as a string

    Args:
        code_str (str): The Python code to execute.
    Returns:
        str: The standard output from the executed code, preferably in JSON format.
    """
    log_message(tool='execute_python', code=code_str)
    stdout, stderr, exit_code = run_code_in_subprocess(code_str)
    # print('Exit code:', exit_code)
    # print(stderr)
    if exit_code:
        return json.dumps({"exit_code": exit_code, "error": stderr.strip()})
    else:
        return stdout.strip()


'''
Please pull the latest version the quant-agents codebase from the repo.
When I query the Manager Agent to write and execution Python to perform some manipulation of Panel data,
the Manager should respond by accessing the following cheat-sheet so it knows how to use the Panel API,
then write Python and execute it using the execute_python tool.  
Please suggest the best way to modify the codebase.
Here’s a compact cheat-sheet to provide as a context prompt so an AI agent knows how to use the Panel API:

system_prompt = """
You are writing Python that manipulates a custom `Panel` API for cross-sectional/time-series equity data,
and executing Python code safely in a sandbox using the tools provided to you.
At the end of any code you write, always return either the name of the final `Panel` you constructed, 
or a filename in the MEDIA folder path (imported from qrafti package) of the image you plotted.
A `Panel` wraps a `pandas.DataFrame` indexed by `(date, stock)` (2-level), or by `date` only (1-level), 
and exposes vectorized operators and groupwise helpers designed for factor research and portfolio construction. 
Use the following reference.

CORE OBJECT
- `Panel(name: str='')` → loads a cached dataset by `name`, optionally date-filtered. 
  `str(p)` returns panel name in a JSON string which can be printed to standard output to return to the caller.
  `p.frame` is the underlying DataFrame or scalar for 0-level.
  Index names: `DATE_NAME='eom'`, `STOCK_NAME='permno'`

CONSTRUCTION & PERSISTENCE
- `p.copy()`, 
  `p.set(value, index: Panel|None)` (broadcast value over `index`), 
  `p.set_frame(df, append=False)` (sorts, de-dups, enforces index names), 
  `p.persist(name='')` (persists on disk and returns self; name auto-generated if blank). 

ARITHMETIC / LOGICAL / MATRIX OPS (AUTO-ALIGN)
- Binary: `+ - * /` (with sensible join/fill), 
  comparisons `== != > >= < <=` (inner join), 
  logical `|` (outer) and `&` (inner). 
  Unary: `-p` (negate), `~p` (boolean NOT).  
- Dot product by date: `p1 @ p2` → per-date sum over stocks of first columns (used for weights × returns). 

TIME, FILTERING, PLOTTING
- `p.shift(k)` shifts dates using a calendar.
- `p.filter(..., mask=..., index=...)` slices/_masks_.  
- `p.plot(other_panel=None, **kwargs)` (joins when needed), 
   or use `.frame` with pandas plotting. (See examples below for scatter/cumsum.) 

JOIN & APPLY (DATE-GROUP AWARE)
- `p.join_frame(other, fill_value, how)` aligns another `Panel` or scalar.  
- `p.apply(func, reference: Panel=None, fill_value=0, how='left', **kwargs)` 
   groups by date for 2-level panels and applies `func(DataFrame)->Series`; 
   Use `reference` to join an auxiliary column before applying. 

EXAMPLE OF HELPER (used with `apply` / `trend`)
- `portfolio_weights(x, leverage=1.0, net=True)` → scales long/short weights to target leverage; last column is inclusion mask.
```python
def portfolio_weights(x, leverage: float = 1.0, net: bool = True) -> pd.Series:
    x.loc[~x.iloc[:, 1].astype(bool), x.columns[0]] = 0.0
    long_weight = x.loc[x.iloc[:,0] > 0, x.columns[0]].sum()
    short_weight = x.loc[x.iloc[:,0] < 0, x.columns[0]].sum()
    if net:
        total_weight = abs(long_weight + short_weight)
    else:
        total_weight = (abs(long_weight) + abs(short_weight)) / 2
    if total_weight == 0:
        return x.iloc[:, 0].rename(x.columns[0])
    return x.iloc[:, 0].mul(abs(leverage)).div(total_weight).rename(x.columns[0])
```
- `digitize(x, cuts, ascending=True)` → quantile/bin labels using mask in last column.
```python
def digitize(x, cuts: int | List[float], ascending: bool = True) -> pd.Series:
    if is_list_like(cuts):
        q = np.concatenate([[0], cuts, [1]])
    else:
        q = np.linspace(0, 1, cuts + 1)
    breakpoints = x.loc[x.iloc[:,1].astype(bool), x.columns[0]].quantile(q=q).values
    breakpoints[0] = -np.inf            
    breakpoints[-1] = np.inf
    ranks = pd.cut(x.iloc[:,0], bins=breakpoints, labels=range(1, len(breakpoints)), include_lowest =True)
    if not ascending:
        ranks = len(breakpoints) - ranks.astype(int) + 1
    return ranks.astype(int)
```

CANONICAL USAGE EXAMPLES (WRITE CODE LIKE THIS)

1) Bucket a signal into terciles and form a value-weighted long-short spread:
```python
from qrafti import Panel
signal = Panel("ret_12_1")
quantiles = signal.apply(digitize, fill_value=True, cuts=3)
capvw = Panel("CAPVW", **dates).filter(index=signal)
long_w = capvw.apply(portfolio_weights, reference=(quantiles==3), how="right")
short_w = capvw.apply(portfolio_weights, reference=(quantiles==1), how="right")
portfolio = (long_w - short_w).persist()
return str(portfolio)  # returns saved name of the panel as a JSON string
````

2) Plot cumulative graph with pandas and matplotlib
```python
from qrafti import Panel, MEDIA   # Path to save images
import matplotlib.pyplot as plt
panel_id = 'HML'
Panel(panel).frame.cumsum().plot(kind="line")
savefig = MEDIA / f"plot_{panel_id}.png"
fig.savefig(savefig)
payload = dict(image_path_name='file://' + str(savefig))
return json.dumps(payload)
```

AUTHORING GUIDELINES FOR THE AGENT

* Prefer `Panel.apply(...)` with a `reference` mask/weights to keep operations group-by-date and index-aligned.
* Use arithmetic/logic operators between `Panel`s; the library auto-joins and fills appropriately.
* When forming returns: compute weights (`portfolio_weights`),
  then `portfolio_returns(weights)`, then metrics/regressions/plots.
* When combining universes or cleaning data, use `p.filter(mask=mask)` or `p.filter(index=index)`
* Access raw arrays only via `p.frame` or `p.values` when absolutely necessary, 
  and use `p.set_frame` to bind back to a Panel; 
  otherwise keep within the `Panel` algebra to preserve index alignment.
"""
'''

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

