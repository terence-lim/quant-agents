"""
Construct residual stock returns from a regression of monthly excess stock returns on excess market returns.
Then define residual volatility as the annualized 12-month volatility of residual stock returns.
Finally, write a research protocol report to evaluate this factor for predicting stock returns.


Define price momentum characteristic as past 12 months' stock returns skipping one month.

Define volatility as the average of squared monthly stock returns over the past 12 months.
"""
import pandas as pd
import numpy as np
import time
from qrafti import write_report, markdown_to_pdf

from qrafti import DATES, rolling

if __name__ == '__main__':
    tic = time.time()
    
    from qrafti import Panel, regression_residuals
    
    if True :
        dates = DATES
        # panel = Panel().load('EXCRET', **dates)
        panel = Panel().load('RET', **dates) - Panel().load('RF', **dates)
        #reference = [Panel().load('Mkt-RF'), Panel().load('HML'), Panel().load('SMB')]
        reference = Panel().load('Mkt-RF')
        ret = panel.trend(regression_residuals, reference)
        print('Elapsed_1', int(time.time() - tic), 'secs')

        window = 12
        skip = 0
        signal = ret.trend(rolling, window=window, skip=skip, agg="std", interval=1)
        description = "Volatility of residual stock returns from regressing monthly stock returns on market returns"
        #signal = Panel().load("ret_12_1", **dates)
        #description = "12-1 Momentum: Prior 12 month returns excluding most recent month"
        print('Elapsed_2', int(time.time() - tic), 'secs')
        

        context = "\n\n".join([description + "\n------------", write_report(signal)])
        with open("output.md", "w") as f:
            f.write(context)
        print('Elapsed_3', int(time.time() - tic), 'secs')
        markdown_to_pdf(context)

        # Sanity plots
        #returns.apply(pd.DataFrame.cumsum).plot(kind="line")
        (-Panel().load("SMB", **dates)).apply(pd.DataFrame.cumsum).plot(kind="line")
        # returns.plot(-Panel().load("SMB", **dates), kind="scatter")
        #plt.show()

        
    if False:
        tic = time.time()
        window, skip = 12, 1
        ret = Panel().load('RET').log1p()
        signal = ret.pow(2).trend(ewm, min_periods=None).pow(0.5)
        print('Elapsed', int(time.time() - tic), 'secs')
        description = "Exponentially-weighted returns volatility"
        context = "\n\n".join([description + "\n------------", write_report(signal)])
        print('Elapsed', int(time.time() - tic), 'secs')
        with open("output.md", "w") as f:
            f.write(context)
        markdown_to_pdf(context)
    
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

    
