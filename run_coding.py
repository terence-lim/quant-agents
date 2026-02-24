'''
Define price momentum characteristic as past 12 months' stock returns skipping one month.

Define volatility as the average of squared monthly stock returns over the past 12 months.

import pandas as pd
import numpy as np
import time

# Let Coding Agent tackle these, with few-shot example from rolling()

def ewm(df: pd.DataFrame, agg: str = "mean", alpha: float = 1 - 0.94, min_periods: int = 0, **kwargs) -> "Panel":
    """Compute an exponentially weighted moving function of this Panel.
    Arguments:
        agg: Aggregation function to apply 'mean' (default), 'sum', 'min', 'max'.
        alpha: Smoothing factor for the EWM, 0 < alpha <= 1
        min_periods: Minimum observations in window required to have a value (if None, then set to half-life)

        **kwargs: additional arguments to pass to pd.DataFrame.ewm.
    Usage:
        panel.trend(ewm, agg="mean", alpha=1-0.94, interval=1)
    """
    if min_periods is None:
        min_periods = int(np.ceil(-np.log(2) / np.log(1 - alpha)))  # half-life implied by alpha
    return df.ewm(alpha=alpha, min_periods=min_periods, **kwargs).agg(agg).where(df.notna())

def expanding(df: pd.DataFrame, agg: str = "mean", **kwargs) -> "Panel":
    """Compute an expanding window function of this Panel.
    Arguments:
        agg: Aggregation function to apply 'mean' (default), 'sum', 'min', 'max'.
        **kwargs: additional arguments to pass to pd.DataFrame.expanding.
    Usage:
        panel.trend(expanding, agg="mean", interval="1")
    """
    return df.expanding(**kwargs).agg(agg).where(df.notna())
'''


code_str =  '''
from qrafti import Panel, DATES
import pandas as pd
import json

def ewm_helper(df: pd.DataFrame, alpha: float, agg: str):
    return df.ewm(alpha=alpha).agg(agg).where(df.notna())
    
stock_returns = Panel().load('RET', **DATES)
squared_returns = stock_returns.pow(2)
result_panel = squared_returns.trend(ewm_helper, agg='mean', alpha=1-0.94).save()
out_dict = result_panel.as_payload()
print(json.dumps(out_dict))
'''

if __name__ == '__main__':
    import time
    tic = time.time()


    if True:
        from server_utils import run_code_in_subprocess
        import json
        stdout, stderr, exit_code = run_code_in_subprocess(code_str)
        # print('Exit code:', exit_code)
        # print(stderr)
        if exit_code:
            out_json = json.dumps({"exit_code": exit_code, "error_message": stderr.strip()})
        else:
            out_json = stdout
        print(out_json)
    
    if False: # True: # rolling regression check 11379 (10 mins for full panel)
        panel = Panel().load('RET') - Panel().load('RF')
        reference = Panel().load('Mkt-RF')
        coeff = 1
        out = panel.trend(rolling_regression, reference, window=12, coeff=coeff, interval=1)

        z = panel[11379].to_frame().join(reference.frame, how='inner', rsuffix='_ref')
        y = rolling_regression(z, window=12, coeff=coeff)

        out[11379]
    if False: # True:  # Calculate Momentum and EWMA volatility: check 11379
        window, skip = 12, 1
        jkp = Panel().load(f'ret_{window}_{skip}')
        ret = Panel().load('RET').log1p()
        mom = ret.trend(rolling, window=window, skip=skip, agg="sum", interval=1).expm1()
        corr = mom.apply(lambda x: x.iloc[:, 0].corr(x.iloc[:, 1], method='pearson'), reference=jkp)
        count = mom.apply(len, reference=jkp, how='inner')
        vol = ret.pow(2).trend(ewm).pow(0.5)
        jkp['2020-12-31':]

    
