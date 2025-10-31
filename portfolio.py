import pandas as pd
import numpy as np
from typing import Dict
'''
class PortfolioEvaluation:
    """Evaluate the performance of a portfolio DataFrame
        if portfolio.nlevels != 2:
            raise ValueError("PortfolioEvaluation requires a PanelFrame with 2 index levels (date, stock)")
        PortfolioEvaluation(portfolio.frame)
    """

    def __init__(self, portfolio: pd.DataFrame):
        if portfolio.nlevels != 2:
            raise ValueError("PortfolioEvaluation requires a PanelFrame with 2 index levels (date, stock)")
        self.portfolio = portfolio

    def turnover(self, ret: PanelFrame) -> PanelFrame:
        """Compute the turnover of the portfolio as the sum of absolute changes in weights.
        Arguments:
            ret: PanelFrame of leading returns to compute drifted portfolio weights
        Returns:
            PanelFrame of turnover values for each date
        """
        # shift both portfolio and returns by 1 period to align
        ret = ret.shift_dates(shift=1)
        shifted_portfolio = self.portfolio.shift_dates(shift=1)

        # left join shifted portfolio with 1 + returns, and multiply to get drifted weights
        df = shifted_portfolio.join_frame(ret + 1, fillna=1, how='left')
        df.iloc[:, 0] = df.iloc[:, 0] * df.iloc[:, 1]  # drift weights by returns
        shifted_portfolio.set_frame(df.iloc[:, [0]])  # update shifted portfolio weights

        # join original portfolio with drifted portfolio weights
        df = self.portfolio.join_frame(shifted_portfolio, fillna=0, how='left')

        # compute turnover as sum of absolute changes in weights
        turnover = df.groupby(level=0).apply(lambda x: (x.iloc[:, 0] - x.iloc[:, -1]).abs().sum())

        return PanelFrame().set_frame(turnover)
    
    
    def information_coefficient(self, ret: PanelFrame) -> PanelFrame:
        """Compute the Information Coefficient (IC) of the factor against the given returns.
        Arguments:
            ret: PanelFrame of returns to compute IC against
        Returns:
            PanelFrame of IC values for each date
        """
        def ic_func(x):
            return x.iloc[:, 0].corr(x.iloc[:, 1])
        return self.portfolio.apply(ic_func, ret, fillna=0)
'''

class PortfolioEvaluation:
    """Evaluate the performance of a portfolio returns DataFrame
    Usage:
        if factor.nlevels != 1:
            raise ValueError("FactorEvaluation requires a DataFrame with single index level (date)")
        FactorEvaluation(factor.frame)    
    """
    def __init__(self, factor: pd.DataFrame, annualization: int = 12):
        self.factor = factor
        self.annualization = annualization

    def annualized_volatility(self) -> float:
        """Compute the annualized volatility of the factor returns."""
        return float(self.factor.std().values[0] * np.sqrt(self.annualization))
    
    def annualized_return(self, geometric: bool = False) -> float:
        """Compute the annualized arithmetic (default) or geometric return of the factor returns."""
        if geometric:
            cum_ret = (1 + self.factor).prod() - 1
            n_periods = len(self.factor)
            avg_ret = (1 + cum_ret.values[0]) ** (self.annualization / n_periods) - 1
        else:
            avg_ret = self.factor.mean().values[0] * self.annualization
        return float(avg_ret)
    
    def sharpe_ratio(self) -> float:
        """Compute the annualized Sharpe ratio of the factor returns."""
        vol = self.annualized_volatility()
        if vol == 0:
            return 0.0
        return float(self.annualized_return() / vol)
    
    def max_drawdown(self) -> float:
        """Compute the maximum drawdown of the factor returns."""
        cum_ret = (1 + self.factor).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        return float(drawdown.min().values[0])

    def summary(self) -> Dict[str, float]:
        """Compute a summary of the factor performance metrics."""
        return {
            'Annualized Return': self.annualized_return(),
            'Volatility': self.annualized_volatility(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Max Drawdown': self.max_drawdown()
        }

