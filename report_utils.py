# report_utils.py    (c) Terence Lim
from qrafti import Panel, DATE_NAME
from utils import plt_savefig, MEDIA
from portfolio import PortfolioEvaluation
from research_utils import digitize, portfolio_weights, portfolio_returns, characteristics_resample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Tuple

#
# Report Writer Function and helpers
#

glossary_md = """
## Explanations:

**Table 1** shows the coverage of the signal for names in the investment universe (US-domiciled common stocks in CRSP).

**Table 2** show the coverage of the signal of total market capitalization of the investment universe.

**Table 3** reports the performance of monthly returns of long-short spread portfolios constructed using a value-weighted tercile sort. The annualized mean return,  annualized volatility, skewness, excess kurtosis, information ratio (=annualized mean divided by annualized volatility), and maximum drawdown are shown.

**Table 4** reports the monthly average excess returns from tercile spread portfolios with and without controlling for common factor models. **Panel A** reports the montly mean return and t-statistic.  **Panel B** reports the alpha and t-statistic relative to the CAPM, along with loading on the market factor. **Panel C** reports the alpha and t-statistic relative to the Fama and French three-factor model, along with the loadings on the market (Mkt-Rf), SMB and HML factors.


**Table 5** presents results for conditional double sorts on size and signal. In each month, stocks are first sorted into quintiles based on size using NYSE breakpoints. Then, within each size quintile, we form equal-weighted tercile spread portfolios based on the signal. The monthly mean return and t-statistic of abnormal returns, as well as the monthly alpha and t-statistic based on the CAPM and Fama and French three-factor models are shown for each size-quintile subset of stocks.

**Figure 1** plots the cumulative returns of the signal adjusted to market beta = 1 (by adding a static long or short position in Mkt-RF so that the full-sample CAPM market beta becomes equal to 1), compared to the cumulative returns on the market factor (Mkt-RF).
"""

def returns_metrics(port_returns: Panel) -> Dict[str, float]:
    """Compute summary performance statistics of portfolio returns.
    Args:
        port_returns: Panel of portfolio returns
    Returns:
        Dict of summary statistics: mean return, volatility, Sharpe ratio
    """
    if port_returns.nlevels != 1:
        return {}
    else:
        return PortfolioEvaluation(returns=port_returns.frame).metrics()

def returns_regression(port_returns: Panel, fac_returns: List[Panel] = []) -> Tuple[dict, pd.Series]:
    """Compute regression coefficients of a portfolio given its returns and other factor returns.
    Arguments:
        port_returns: Panel of portfolio returns
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

    context = []

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
    if len(eom) == 1 or eom[0] > 6 * eom[1]:
        signal = characteristics_resample(signal, month=int(eom.index[0]))
        print(f"Signal resampled to month end of month {eom.index[0]}")  ###

    ### TODO: inner join to SIZE_DECILE universe
        
    # Coverage of count
    name = '% of Names'
    coverage = _compute_coverage(signal.ones_like(), Panel().load("TOTAL_COUNT"))
    coverage = _group_coverage(coverage, name)
    context.append(f"### Table 1. {name} Covered by Period\n" + coverage.round(2).to_markdown())

    # Coverage of cap
    name = '% of Market Cap'
    cap = Panel().load("CAP").restrict(subset=signal)
    coverage = _compute_coverage(cap, Panel().load("TOTAL_CAP"))
    coverage = _group_coverage(coverage, name)
    context.append(f"### Table 2. {name} Covered by Period\n" + coverage.round(2).to_markdown())

    # Form portfolio weights
    quantiles = signal.apply(digitize, fill_value=True, cuts=3)
    capvw = Panel().load("CAP").restrict(subset=signal)
    q3 = capvw.apply(portfolio_weights, reference=quantiles == 3) #, how="right")
    q1 = capvw.apply(portfolio_weights, reference=quantiles == 1) #, how="right")
    portfolio = q3 - q1

    # TODO: turnover
    # drifted = portfolio_impute(portfolio, drifted=True)
    # trades = portfolio.restrict(subset=drifted) - drifted
    # turnover = trades.apply(pd.DataFrame.abs).apply(pd.DataFrame.sum)/2

    # Form portfolio returns
    returns = portfolio_returns(portfolio)
    stats = returns_metrics(returns)
    df = pd.Series(stats, name="High minus Low").to_frame().T
    context.append(
        "### Table 3. Statistics of Tercile Spread Portfolios\n(weighted by market cap)"
    )
    context.append(df.round(4).to_markdown())

    # by model
    context.append("### Table 4. Alpha, coefficients and t-statistics by Model")
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
    if savefig:  # Plot cumulative returns
        # Plot cumulative Mkt-RF for comparison
        mkt_rf = Panel().load("Mkt-RF").restrict(subset=returns)
        # Adjust returns by subtracting beta * Mkt-RF to isolate alpha performance
        adjusted_returns = returns.frame.iloc[:, 0] + (1-beta) * mkt_rf.frame.iloc[:, 0]
        cumulative_returns = adjusted_returns.cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns.index, cumulative_returns,
                 label='Tercile Return (adjusted to Beta=1)', color='blue')
        cumulative_mkt_rf = mkt_rf.frame.cumsum()
        plt.plot(cumulative_mkt_rf.index, cumulative_mkt_rf.iloc[:, 0], 
                 label='Cumulative Mkt-RF', linestyle='--')  
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Returns of Tercile Spread Portfolio')
        plt.legend()
        plt.tight_layout()
        plt_savefig(savefig)
        plt.close()

    ff3,_ = returns_regression(returns, [Panel().load("Mkt-RF"), Panel().load("SMB"), Panel().load("HML")])
    df = pd.DataFrame(
        {
            "coefficients": {"intercept": ff3["intercept"]} | ff3["coefficients"],
            "t-stats": {"intercept": ff3["t_intercept"]} | ff3["t_statistics"],
        }
    ).rename_axis(index="Fama-French 3-Factor Model")
    context.append(df.round(4).to_markdown())

    # Evaluate alphas by size quintile
    size_decile = Panel().load("SIZE_DECILE").restrict(subset=signal)
    out = []
    for quintile, sz in enumerate([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]):
        size_mask = size_decile.apply(pd.DataFrame.isin, values=sz)
        quantiles_sz = signal.apply(digitize, size_mask, cuts=3)
        high_sz = (quantiles_sz == 3).apply(portfolio_weights, fill_value=True)
        low_sz = (quantiles_sz == 1).apply(portfolio_weights, fill_value=True)
        portfolio_sz = high_sz - low_sz
        returns_sz = portfolio_returns(portfolio_sz)
        mu_sz, _ = returns_regression(returns_sz)
        capm_sz, _ = returns_regression(returns_sz, [Panel().load("Mkt-RF")])
        ff3_sz, _ = returns_regression(
            returns_sz, [Panel().load("Mkt-RF"), Panel().load("SMB"), Panel().load("HML")]
        )
        out.append(
            pd.DataFrame(
                [
                    mu_sz["intercept"],
                    mu_sz["t_intercept"],
                    capm_sz["intercept"],
                    capm_sz["t_intercept"],
                    ff3_sz["intercept"],
                    ff3_sz["t_intercept"],
                ],
                index=[
                    "mean",
                    "t-stat",
                    "alpha (CAPM)",
                    "t-stat (CAPM)",
                    "alpha (FF3)",
                    "t-stat (FF3)",
                ],
                columns=[f"Size Quintile {quintile + 1}"],
            )
        )
    df = pd.concat(out, axis=1).rename_axis(index="Model")
    context.append(
        "### Table 5. Alpha and t-statistics by Model and Size Quintile\n(lower quintiles have smaller market cap)"
    )
    context.append(df.round(4).to_markdown())

    context = "\n\n".join(context)
    return context

