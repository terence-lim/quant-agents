# report_utils.py    (c) Terence Lim
from qrafti import Panel, DATE_NAME
from utils import plt_savefig, MEDIA
from portfolio import PortfolioEvaluation
from research_utils import digitize, portfolio_weights, portfolio_returns, characteristics_resample, portfolio_impute
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Tuple

SIZE_DECILE = "size_decile"
TOTAL_COUNT = "total_count"
TOTAL_CAP = "total_cap"
CAP = "mthcap"

#
# Report Writer Function and helpers
#

glossary_md = """
## Explanations:

**Table 1** shows the coverage of the signal for names in and market capitalization of the investment universe (US-domiciled common stocks in CRSP).

**Table 2** shows the monthly and annual turnover and estimate trading costs for long-short spread portfolios constructed using a value-weighted tercile sort.

**Table 3** reports the performance of monthly returns of the spread portfolios. The annualized mean return,  annualized volatility, skewness, excess kurtosis, sharpe ratio (=annualized mean divided by annualized volatility), and maximum drawdown are shown.

**Table 4** reports the monthly average excess returns from tercile spread portfolios with and without controlling for common factor models. **Panel A** reports the montly mean return and t-statistic.  **Panel B** reports the alpha and t-statistic relative to the CAPM, along with loading on the market factor. **Panel C** reports the alpha and t-statistic relative to the Fama and French three-factor model, along with the loadings on the market (Mkt-Rf), SMB and HML factors.


**Table 5** presents results for conditional double sorts on size and signal. In each month, stocks are first sorted into quintiles based on size using NYSE breakpoints. Then, within each size quintile, we form equal-weighted tercile spread portfolios based on the signal. The monthly mean return and t-statistic of abnormal returns, as well as the monthly alpha and t-statistic based on the CAPM and Fama and French three-factor models are shown for each size-quintile subset of stocks.

**Figure 1** plots the cumulative returns of the signal adjusted to market beta = 1 (by adding a static long or short position in Mkt-RF so that the full-sample CAPM market beta becomes equal to 1), compared to the cumulative returns on the market factor (Mkt-RF).
"""

def returns_regression(port_returns: Panel, fac_returns: List[Panel] = []) -> Tuple[dict, pd.Series]:
    """Compute regression coefficients of a portfolio given its excess returns and other factor returns.
    Arguments:
        port_returns: Panel of excess portfolio returns
        fac_returns: List of Panels of factor returns
    Returns:
        Tuple(Dict of coefficients and regression statistics, Series of residual returns)
    """
    if port_returns.nlevels != 1:
        return ({}, None)
    factor_frames = [factor.frame for factor in fac_returns if factor.nlevels == 1]
    return PortfolioEvaluation(returns=port_returns.frame).regression(factor_frames)

def write_report(signal: Panel, savefig: str = MEDIA / 'output.png') -> str:
    """Compute factor returns and performance statistics from stock characteristics
    Arguments:
        signal: Panel of stock characteristic values from which to calculate and evaluate factor returns
    Returns:
        str: Evaluation results and tables in markdown format
    """

    def _compute_coverage(num, den):
        """Compute row coverage as the grouped sum of `num` divided by `den`"""
        start_date = max(min(den.dates), min(num.dates))
        end_date = min(max(den.dates), max(num.dates))
        return (num.restrict(start_date=start_date, end_date=end_date).frame.groupby(DATE_NAME).sum().iloc[:,0] /
                den.restrict(start_date=start_date, end_date=end_date).frame.iloc[:,0])

    def _group_coverage(df, col, max_years=6, num_subperiods=3):
        """Groups a DataFrame by year or subperiods based on the number of unique years"""
        df = df.rename(name).reset_index()
        df["year"] = df[DATE_NAME].dt.year
        
        # 1. Get the unique sorted years
        years = sorted(df["year"].unique())
        num_years = len(years)

        if num_years >= max_years:
            # 2. Split years into 3 arrays (handles uneven counts automatically)
            subperiod_arrays = np.array_split(years, num_subperiods)
            
            # 3. Create a mapping of year -> "Start-End" string
            period_map = {}
            for arr in subperiod_arrays:
                label = f"{arr[0]}-{arr[-1]}"
                for y in arr:
                    period_map[y] = label
            
            # 4. Map the years to their subperiod labels
            df["year"] = df["year"].map(period_map)
        return df.groupby("year")[col].mean().to_frame(name=col) * 100

    # check if signal is mostly annual data
    eom = signal.frame.index.get_level_values(DATE_NAME).strftime('%m').value_counts().astype(int)
    if len(eom) == 1 or eom.iloc[0] > 6 * eom.iloc[1]:
        signal = characteristics_resample(signal, month=int(eom.index[0]))
        print(f"Signal resampled to month end of month {eom.index[0]}")  ###

    # inner join to SIZE_DECILE universe
    signal = signal.restrict(subset=Panel().load(SIZE_DECILE))

    context = []

    # Coverage of count
    name = '% of Names'
    count_coverage = _compute_coverage(signal.ones_like(), Panel().load(TOTAL_COUNT))
    count_coverage = _group_coverage(count_coverage, name)
    # context.append(f"### Table 1. {name} Covered by Period\n" + count_coverage.round(2).to_markdown())

    # Coverage of cap
    name = '% of Market Cap'
    cap = Panel().load(CAP).restrict(subset=signal)
    cap_coverage = _compute_coverage(cap, Panel().load(TOTAL_CAP))
    cap_coverage = _group_coverage(cap_coverage, name)
    # context.append(f"### Table 2. {name} Covered by Period\n" + cap_coverage.round(2).to_markdown())

    coverage = pd.concat([count_coverage, cap_coverage], axis=1)
    context.append(f"### Table 1. Coverage of Investment Universe by Period\n" + coverage.round(2).to_markdown())
    
    # Form portfolio weights
    quantiles = signal.apply(digitize, fill_value=True, cuts=3)
    capvw = Panel().load(CAP).restrict(subset=signal)
    q3 = capvw.apply(portfolio_weights, reference=quantiles == 3) #, how="right")
    q1 = capvw.apply(portfolio_weights, reference=quantiles == 1) #, how="right")
    portfolio = q3 - q1

    # TODO: add turnover to report
    drifted = portfolio_impute(portfolio, drifted=True)
    trades = portfolio.restrict(subset=drifted) - drifted
    turnover = trades.frame.abs().groupby(DATE_NAME).sum()
    pct_turnover = (turnover / portfolio.frame.abs().groupby(DATE_NAME).sum()).dropna()
    df = pd.DataFrame({'Turnover': [pct_turnover.mean().iloc[0], 12 * pct_turnover.mean().iloc[0]],
                       'T-cost (@5bps)': [turnover.mean().iloc[0] * 0.0005, 12 * turnover.mean().iloc[0] * 0.0005],
                       'T-cost (@10bps)': [turnover.mean().iloc[0] * 0.001, 12 * turnover.mean().iloc[0] * 0.001]},
                      index=['Monthly', 'Annual'])
    context.append(f"### Table 2. Portfolio Turnover and T-cost\n" + df.round(4).to_markdown())
                             
                             
    # Form portfolio returns
    returns = portfolio_returns(portfolio)
    stats = PortfolioEvaluation(returns=returns.frame).annualized_metrics(digits=4)
    df = pd.DataFrame(stats, index=["High minus Low"]).iloc[:, :-2]
    df.index.name = "Annualized"
    context.append(
        "### Table 3. Statistics of Tercile Spread Portfolios\n(weighted by market cap)"
    )
    context.append(df.round(4).to_markdown())

    # by model
    context.append("### Table 4. Monthly alpha, coefficients and t-statistics by Model")
    mu,_ = returns_regression(returns, [])
    df = pd.DataFrame(
        {
            "coefficients": {"intercept": mu["intercept"]} | mu["coefficients"],
            "t-stats": {"intercept": mu["t_intercept"]} | mu["t_statistics"],
        }
    ).rename_axis(index="Mean Returns")
    context.append(df.round(4).to_markdown())

    capm,_ = returns_regression(returns, [Panel().load("Mkt-RF")])
    df = pd.DataFrame(
        {
            "coefficients": {"intercept": capm["intercept"]} | capm["coefficients"],
            "t-stats": {"intercept": capm["t_intercept"]} | capm["t_statistics"],
        }
    ).rename_axis(index="CAPM")
    context.append(df.round(4).to_markdown())

    beta = capm["coefficients"].get("Mkt-RF", 0)

    ff3,_ = returns_regression(returns, [Panel().load("Mkt-RF"), Panel().load("SMB"), Panel().load("HML")])
    df = pd.DataFrame(
        {
            "coefficients": {"intercept": ff3["intercept"]} | ff3["coefficients"],
            "t-stats": {"intercept": ff3["t_intercept"]} | ff3["t_statistics"],
        }
    ).rename_axis(index="Fama-French 3-Factor Model")
    context.append(df.round(4).to_markdown())

    if savefig:  # Plot cumulative returns
        # Plot cumulative Mkt-RF for comparison
        ### mkt_rf = Panel().load("Mkt-RF").restrict(subset=returns)
        # Adjust returns by subtracting beta * Mkt-RF to isolate alpha performance
        ### adjusted_returns = returns.frame.iloc[:, 0] + (1-beta) * mkt_rf.frame.iloc[:, 0]
        ### cumulative_returns = adjusted_returns.cumsum()
        cumulative_returns = returns.frame.iloc[:, 0].cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns.index, cumulative_returns, color='blue',
                 label='Tercile Residual Returns (FF3-adjusted)')
        ### label='Tercile Return (adjusted to Beta=1)')
        ### cumulative_mkt_rf = mkt_rf.frame.cumsum()
        ### plt.plot(cumulative_mkt_rf.index, cumulative_mkt_rf.iloc[:, 0], 
        ###         label='Cumulative Mkt-RF', linestyle='--')  
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Returns of Tercile Spread Portfolio')
        plt.legend()
        plt.tight_layout()
        plt_savefig(savefig)
        plt.close()

    # Evaluate alphas by size quintile
    size_decile = Panel().load(SIZE_DECILE).restrict(subset=signal)
    out = []
    for quintile, sz in enumerate([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]):
        size_mask = size_decile.apply(pd.DataFrame.isin, values=sz)
        signal_sz = signal.restrict(mask=size_mask)        
        quantiles_sz = signal_sz.apply(digitize, cuts=3, fill_value=True)
        high_sz = (quantiles_sz == 3).apply(portfolio_weights, fill_value=True)
        low_sz = (quantiles_sz == 1).apply(portfolio_weights, fill_value=True)
        num_stocks = signal_sz.frame.groupby(DATE_NAME).count()
        portfolio_sz = high_sz - low_sz
        returns_sz = portfolio_returns(portfolio_sz)
        mu_sz, _ = returns_regression(returns_sz)
        capm_sz, _ = returns_regression(returns_sz, [Panel().load("Mkt-RF")])
        ff3_sz, _ = returns_regression(returns_sz,
                                       [Panel().load("Mkt-RF"), Panel().load("SMB"), Panel().load("HML")])
        out.append(pd.DataFrame([num_stocks.mean().iloc[0].round(1),
                                 mu_sz["intercept"], mu_sz["t_intercept"],
                                 capm_sz["intercept"], capm_sz["t_intercept"],
                                 ff3_sz["intercept"], ff3_sz["t_intercept"]],
                   index=["Num Stocks", "mean", "t-stat", "alpha (CAPM)", "t-stat (CAPM)", "alpha (FF3)", "t-stat (FF3)"],
                   columns=[f"Size Quintile {quintile + 1}"]))

    df = pd.concat(out, axis=1).rename_axis(index="Model")
    context.append(
        "### Table 5. Monthly alpha and t-statistics by Model and Size Quintile\n(lower quintiles have larger market cap)"
    )
    context.append(df.round(4).to_markdown())

    context = "\n\n".join(context)
    return context

if __name__ == "__main__":
    signal = Panel().load('_1')
    savefig = MEDIA / 'output.png'
    output = write_report(signal, savefig)
