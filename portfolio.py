# portfolio.py    (c) Terence Lim
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import statsmodels.api as sm

class PortfolioEvaluation:
    """Evaluate the performance of a portfolio returns DataFrame
    Usage:
        if portfolio.nlevels != 1:
            raise ValueError("PortfolioEvaluation requires a DataFrame with single index level (date)")
        PortfolioEvaluation(portfolio.frame)    
    """
    def __init__(self, returns: pd.DataFrame, annualization: int = 12):
        self.returns = returns
        self.annualization = annualization

    def annualized_volatility(self) -> float:
        """Compute the annualized volatility of the portfolio returns."""
        return float(self.returns.std().values[0] * np.sqrt(self.annualization))
    
    def annualized_return(self, geometric: bool = False) -> float:
        """Compute the annualized arithmetic (default) or geometric return of the portfolio returns."""
        if geometric:
            cum_ret = (1 + self.returns).prod() - 1
            n_periods = len(self.returns)
            avg_ret = (1 + cum_ret.values[0]) ** (self.annualization / n_periods) - 1
        else:
            avg_ret = self.returns.mean().values[0] * self.annualization
        return float(avg_ret)
    
    def information_ratio(self) -> float:
        """Compute the annualized information ratio of the portfolio returns."""
        vol = self.annualized_volatility()
        if vol == 0:
            return 0.0
        return float(self.annualized_return() / vol)
    
    def max_drawdown(self) -> float:
        """Compute the maximum drawdown of the portfolio returns."""
        cum_ret = (1 + self.returns).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        return float(drawdown.min().values[0])

    def skewness(self) -> float:
        """Compute the skewness of the portfolio returns"""
        return float(self.returns.skew().values[0])

    def excess_kurtosis(self) -> float:
        """Compute the excess kurtosis of the portfolio returns"""
        return float(self.returns.kurt().values[0])

    def regression(self, bench: Optional[List[pd.DataFrame]] = None) -> Tuple[dict, pd.Series]:
        """Run one multivariate OLS: portfolio ~ const + all benchmarks (independent vars)."""
        bench = bench or []

        # y: portfolio returns (first/only column)
        y = self.returns.iloc[:, 0].astype(float)

        # Build X from each benchmark's first column; keep names if present
        X_parts = []
        for i, b in enumerate(bench):
            s = b.squeeze()  # support DataFrame or Series
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            if s.name is None:
                s = s.rename(f"bench_{i}")
            X_parts.append(s.astype(float))

        X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=y.index)

        # Align y and X, drop missing
        df = pd.concat([y.rename("factor"), X], axis=1).dropna()
        y = df["factor"]
        X = df.drop(columns=["factor"])

        # Add intercept and fit OLS
        X = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, X).fit()

        # Extract params and t-stats
        params = res.params
        tvals = res.tvalues

        intercept = float(params["const"])
        t_intercept = float(tvals["const"])

        coef_names = [c for c in params.index if c != "const"]
        coefficients = {name: float(params[name]) for name in coef_names}
        t_statistics = {name: float(tvals[name]) for name in coef_names}

        return {
            "intercept": intercept,
            "t_intercept": t_intercept,
            "coefficients": coefficients,
            "t_statistics": t_statistics,
            "resid_variance": float(res.mse_resid),
            "n_obs": int(res.nobs),
            "r2": float(res.rsquared),
            "adj_r2": float(res.rsquared_adj),
        }, res.resid

    def metrics(self) -> Dict[str, float]:
        """Compute key portfolio returns performance metrics."""
        return {
            'Annualized Return': self.annualized_return(),
            'Volatility': self.annualized_volatility(),
            'Skewness': self.skewness(),
            'Excess Kurtosis': self.excess_kurtosis(),
            'Information Ratio': self.information_ratio(),
            'Max Drawdown': self.max_drawdown(),
            'Num Obs': len(self.returns),
            'Start Date': self.returns.index[0].strftime('%Y-%m-%d'),
            'End Date': self.returns.index[-1].strftime('%Y-%m-%d'),
        }

