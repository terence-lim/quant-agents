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

#
# Example of using common libraries such as matplotlib to operating on the underlying pandas DataFrame
#
from qrafti import Panel, DATES, plt_savefig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Load the data panel for returns and select date range, and extract its DataFrame
panel_id = 'HML'
returns_panel = Panel().load(panel_id, **DATES).frame
# Use matplotlib to plot cumulative returns
plt.plot(returns_panel.index, returns_panel.cumsum().values)
plt.title(panel_id)
plt.xlabel('Date')
plt.ylabel('Return')
plt.tight_layout()
# Save the image and return its filename in json dictionary format to stdout
out_dict = {"image file name": plt_savefig()}
print(out_dict)

#
# Example of winsoring cross-sections of Panel data, by date across stocks
#
def winsorize_helper(x, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """
    Winsorize the first column based on the quantiles of the true rows in the last column.
    Arguments:
        x: DataFrame with at least two columns, first column is the data to be winsorized;
           last column is a boolean indicator for which rows to consider for winsorizing
        lower: lower quantile to use for winsorizing (default 0.05)
        upper: upper quantile to use for winsorizing (default 0.95)
    Returns:
        pd.Series with the winsorized values of the first column
    """
    if x.shape[1] > 1:  # if there is an indicator column, use it to determine which rows to consider for quantiles
        lower, upper = (
            x.loc[x.iloc[:, -1].astype(bool)].iloc[:, 0].quantile([lower, upper]).values
        )
    else:
        lower, upper = x.iloc[:, 0].quantile([lower, upper]).values
    return x.iloc[:, 0].clip(lower=lower, upper=upper)

# Load the data panel for returns, select date range, and optionally load an indicator panel
panel_id = 'RET'
data_panel = Panel().load(panel_id, **DATES)
indicator_panel_id = 'EXCHCD'
if indicator_panel_id:
    indicator_panel = Panel().load(indicator_panel_id, **DATES) == 1  # indicator: exchange code equals 1
# Apply winsorization to the data panel using the helper function, optionally with the indicator panel, and save the result.
how = "left"    # how to align the indicator panel with the data panel: 'left' (default), 'inner', 'right'
fill_value = 0  # value to fill for missing indicator values when aligning panels
result_panel = data_panel.apply(winsorize_helper,
                                indicator_panel if indicator_panel_id else None,
                                how=how,
                                fill_value=fill_value).save()
# Return ID of the resulting panel in a dictionary json format.
out_dict = result_panel.as_payload()  

#
# Example of computing time series regression residuals in Panel data, by stock across time
#
def residuals_helper(x: pd.DataFrame) -> pd.Series:
    """Compute residuals from OLS regression of y ~ 1 + x1 + x2 + ...
    Arguments:
        x: DataFrame with columns 'y', 'x1', 'x2', ...
    Returns:
        pd.Series of residuals, indexed the same as x
    """
    def _ols_residuals(y, X) -> np.ndarray:
        """OLS regression: y ~ 1 + X"""
        X = np.column_stack([np.ones(len(X)), X])   # add intercept
        if np.isfinite(X).all() and np.isfinite(y).all():
            try:
                betas, *_ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                betas = np.linalg.pinv(X) @ y  # fallback to pseudo-inverse
            return y - X @ betas
        else:
            return None    
    residuals = _ols_residuals(x.iloc[:, 0].values, x.iloc[:, 1 :].values)
    if residuals is None:
        return pd.Series([np.nan] * len(x), index=x.index)
    else:
        return pd.Series(residuals, index=x.index)

# Load returns panels and select date range.
panel_id = 'RET'
other_panel_ids = ['Mkt-RF', 'SMB', 'HML']
returns_panel = Panel().load(panel_id, **DATES)
other_panels = [Panel().load(pid, **DATES) for pid in other_panel_ids]
# Compute residuals from regressions of time series returns on factors and save the result.
result_panel = returns_panel.trend(residuals_helper, other_panels).save()  
# Return ID of the resulting panel in a dictionary json format.
out_dict = result_panel.as_payload()  
print(out_dict)

#
# Example using pandas builtin rolling method to compute time-series statistics of Panel data, by stock over time.
#
def rolling_helper(df: pd.DataFrame) -> pd.Series:
    """Helper to apply a rolling window aggregation function to a DataFrame
    Arguments:
        df: DataFrame of log returns for a single stock, indexed by date
    Returns:
        pd.Series with the rolling aggregated values, aligned with the original index
    """
    window = 12   # 12 months
    skip = 1      # skip the most recent month
    agg = "sum"   # sum of log returns over the window
    return df.shift(periods=skip).rolling(window=window-skip).agg(agg).where(df.notna())

# Load returns panel, select date range, and compute log returns.
panel_id = 'RET'
log_returns = Panel().load(panel_id, **DATES).log1p()
 # Apply the rolling helper function to compute price momentum on time series of stock returns and save the result.
result_panel = log_returns.trend(rolling_helper).save()
# Return ID of the resulting panel in a dictionary json format.
out_dict = result_panel.as_payload()  
print(out_dict)



code_str = '''
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

    if False:
        from server_utils import run_code_in_subprocess
        import json
        
        stdout, stderr, exit_code = run_code_in_subprocess(code_str)

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

    
