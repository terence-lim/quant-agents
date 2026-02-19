from qrafti import Panel, Calendar, STOCK_NAME, DATE_NAME
import pandas as pd
import numpy as np
from typing import Callable


import traceback
panel_id = 'RET'
panel_id = '_100'
#panel_id = 1
window = 12
skip = 1
interval = 1
agg = "sum"
try:
    p1 = panel_or_numeric(panel_id, **DATES)
    p2 = p1.trend(rolling, window=window, skip=skip, agg=agg, interval=interval)
    p3 = p2
    out = p3.as_payload()
    log_tool(tool="Panel_characteristics_rolling",
             input=dict(panel_id=panel_id, window=window, skip=skip, agg=agg, interval=interval),
             output=out)    
except Exception as e:
    out = dict(error=traceback.format_exc())

print(json.dumps(out))
    #    print(type(e))
    #    print(f"Error: {e}")
    #    print(e.args)

class P:
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            print(start, stop)
        return None
        #     keys = [k for k in self.data if (start is None or k >= start)
        #                                    and (stop  is None or k <= stop)]
        #    return {k: self.data[k] for k in keys}

if False:
    skip = 1
    window = 12
    time_series = panel[11379].to_frame()
    time_series['permno'] = 11379
    time_series = time_series[['permno', 'RET']]





panel = Panel('LOG1P_RET')

def momentum(df: pd.DataFrame, window: int = 12, skip: int = 1) -> pd.DataFrame:
    """rolling sum of past `window` months skipping the most recent `skip` months"""
    return df.shift(skip).rolling(window=window-skip).sum().where(df.notna())
mom = panel.trend(momentum)
mom[11379]

mom1 = panel.rolling(window=12, skip=1, agg="sum")
mom1[11379]


def ewma_volatility(df: pd.DataFrame, alpha: float = 1 - 0.94) -> pd.Series:
    return df.pow(2).ewm(alpha=alpha).agg("mean").pow(0.5).where(df.notna(), np.nan)
vol = panel.trend(ewma_volatility)
vol[11379]

vol1 = panel.pow(2).ewm(alpha=1-0.94).pow(0.5)
vol1[11379]

vol2 = panel.pow(2).expanding().pow(0.5)
vol2[11379]




# EWMA of squared returns
# panel[11379]
df = pd.DataFrame(
    {"RET": [0.02, -0.01, 0.015, 0.03, -0.005]},
    index=pd.date_range("2020-01-31", periods=5, freq="ME"),
)
lambda_ = 0.94  # RiskMetrics recommended for monthly data
df["RET"].pow(2).ewm(alpha=1 - lambda_).agg("mean")
