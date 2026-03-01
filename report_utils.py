from qrafti import Panel, DATE_NAME, STOCK_NAME
from utils import plt_savefig, MEDIA
from portfolio import PortfolioEvaluation
from research_utils import digitize, portfolio_weights, portfolio_returns
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Tuple

#
# Report Writer Function and helpers
#

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
            # print(years)  ###
            # print(subperiod_arrays)  ###
            
            # 3. Create a mapping of year -> "Start-End" string
            period_map = {}
            for arr in subperiod_arrays:
                label = f"{arr[0]}-{arr[-1]}"
                for y in arr:
                    period_map[y] = label
            
            # 4. Map the years to their subperiod labels
            # print(df) ###
            df["year"] = df["year"].map(period_map)
        return df.groupby("year")[col].mean().to_frame(name=col) * 100

    # Coverage of count
    name = '% of Names'
    coverage = _compute_coverage(signal.ones_like(), Panel().load("TOTAL_COUNT"))
    coverage = _group_coverage(coverage, name)
    context.append(f"### {name} Covered by Period\n" + coverage.round(2).to_markdown())

    # Coverage of cap
    name = '% of Market Cap'
    cap = Panel().load("CAP").restrict(subset=signal)
    coverage = _compute_coverage(cap, Panel().load("TOTAL_CAP"))
    coverage = _group_coverage(coverage, name)
    context.append(f"### {name} Covered by Period\n" + coverage.round(2).to_markdown())

    # Form portfolios
    # print('signal', signal.frame)  ###
    quantiles = signal.apply(digitize, fill_value=True, cuts=3)
    capvw = Panel().load("CAP").restrict(subset=signal)
    # print('capvw', capvw.frame)  ###
    q3 = capvw.apply(portfolio_weights, reference=quantiles == 3) #, how="right")
    q1 = capvw.apply(portfolio_weights, reference=quantiles == 1) #, how="right")
    portfolio = q3 - q1
    # print('portfolio', portfolio.frame)  ###

    # turnover
    # drifted = portfolio_impute(portfolio, drifted=True)
    # trades = portfolio.restrict(subset=drifted) - drifted
    # turnover = trades.apply(pd.DataFrame.abs).apply(pd.DataFrame.sum)/2

    # Evaluate returns
    returns = portfolio_returns(portfolio)
    if savefig:
        # Plot cumulative returns
        cumulative_returns = returns.frame.cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns.index, cumulative_returns.iloc[:, 0], label='Cumulative Return')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Returns of Tercile Spread Portfolio')
        plt.tight_layout()
        plt_savefig(savefig)
        plt.close()
    stats = returns_metrics(returns)
    df = pd.Series(stats, name="High minus Low").to_frame().T
    context.append(
        "### Statistics of Tercile Spread Portfolios\n(weighted by market cap winsorized at 80th NYSE percentile)"
    )
    context.append(df.round(4).to_markdown())

    # by model
    context.append("### Alpha, coefficients and t-statistics by Model")
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
        "### Alpha and t-statistics by Model and Size Quintile\n(lower quintiles have smaller market cap)"
    )
    context.append(df.round(4).to_markdown())

    context = "\n\n".join(context)
    return context

