from qrafti import Panel, DATE_NAME, STOCK_NAME, CRSP_VERSION
from utils import Calendar
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Tuple
from pandas.api.types import is_list_like
from tqdm import tqdm
#
# Common tools to applied on cross-sectional slices with Panel.apply()
#
def winsorize(x, lower=0.0, upper=1.0) -> pd.Series:
    """
    Winsorize the first column based on the quantiles of the true rows in the last column.
    Arguments:
        x: DataFrame with at least two columns, first column is the data to be winsorized,
           last column is a boolean indicator for which rows to consider for winsorizing
        lower: Lower quantile threshold (between 0 and 1)
        upper: Upper quantile threshold (between 0 and 1)
    Returns:
        pd.Series with the winsorized values of the first column
    Usage:
        panel_frame.apply(winsorize, indicator or True, fill_value=False, lower=lower, upper=upper)
    """
    lower, upper = (
        x.loc[x.iloc[:, 1].astype(bool)].iloc[:, 0].quantile([lower, upper]).values
    )
    return x.iloc[:, 0].clip(lower=lower, upper=upper)


def digitize(x, cuts: int | List[float], ascending: bool = True) -> pd.Series:
    """
    Discretize values into bins based on quantiles calculated from a filtered subset of the data.

    This function calculates quantile breakpoints using only the rows where the second column 
    is True. It then applies these breakpoints to categorize every row in the first column 
    into discrete bin numbers.

    ### Logic:
    1. **Breakpoint Calculation**: Quantiles are determined from `x.iloc[:, 0]` but ONLY for 
       rows where `x.iloc[:, 1]` is True.
    2. **Binning**: All values in `x.iloc[:, 0]` are then mapped into these bins.
    3. **Ranking**: Bin 1 contains the lowest values (if ascending=True).

    Args:
        x (pd.DataFrame): DataFrame where:
            - Column 0: The data to be binned.
            - Column 1: A boolean/indicator mask used to select the "training" data 
              for calculating quantile breakpoints.
        cuts (int | List[float]): 
            - If `int`: Number of equal-width quantiles (e.g., 5 for quintiles).
            - If `List[float]`: Specific quantile probabilities excluding endpoints (e.g., [0.33, 0.66]).
        ascending (bool): Defaults to True. 
            - If True, bin 1 is the lowest value group. 
            - If False, bin 1 is the highest value group.

    Returns:
        pd.Series: Integer labels starting from 1 representing the bin assignment for each row.

    Usage:
        # Categorize a factor into 5 bins using only 'Liquid' stocks to define the deciles:
        panel.apply(digitize, cuts=5)
    """
    if is_list_like(cuts):
        q = np.concatenate([[0], cuts, [1]])
    else:
        q = np.linspace(0, 1, cuts + 1)
    breakpoints = x.loc[x.iloc[:, 1].astype(bool), x.columns[0]].quantile(q=q).values
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    ranks = pd.cut(
        x.iloc[:, 0],
        bins=breakpoints,
        labels=range(1, len(breakpoints)),
        include_lowest=True,
    )
    if not ascending:
        ranks = len(breakpoints) - ranks.astype(int) + 1
    return ranks.astype(int)


def portfolio_weights(x) -> pd.Series:
    """Scale the the portfolio weights to sum 1.0
    Arguments:
        x: DataFrame with at least two columns, first column is the raw unscaled weights,
           last column is a boolean indicator for which rows to keep in the portfolio
    Returns:
        pd.Series with the scaled weights
    Usage:
        panel_frame.apply(portfolio_weights)
    """
    # set weights to zero for rows where second column is False
    x.loc[~x.iloc[:, 1].astype(bool), x.columns[0]] = 0.0
    long_weight = x.loc[x.iloc[:, 0] > 0, x.columns[0]].sum()
    short_weight = x.loc[x.iloc[:, 0] < 0, x.columns[0]].sum()
    if abs(long_weight) < 1e-6 and abs(short_weight) < 1e-6:
        total_weight = (abs(long_weight) + abs(short_weight)) / 2
    else:   # long-only or short-only portfolio
        total_weight = abs(long_weight) + abs(short_weight)
    if total_weight == 0:
        return x.iloc[:, 0].rename(x.columns[0])
    return x.iloc[:, 0].div(total_weight).rename(x.columns[0])

#
# Common functions to be applied on time-series slices with Panel.trend()
#

def rolling(df: pd.DataFrame, window: int, skip: int = 0, agg: str = "mean", **kwargs) -> pd.Series:
    """Apply a rolling window aggrgation function to a DataFrame.
    Arguments:
        window: Size of the rolling window, min_periods will default to this integer value.
        skip: Number of periods at the end of the window to skip (default is 0).
        agg: Aggregation function to apply 'mean' (default), 'sum', 'min', 'max'.
        **kwargs: additional arguments to pass to pd.DataFrame.rolling.
    Usage:
        panel.trend(rolling, window=12, skip=1, agg="mean", interval=1)
    
    """
    return df.shift(periods=skip).rolling(window=window-skip, **kwargs).agg(agg).where(df.notna())

def rolling_regression(x: pd.DataFrame, window: int, coeff: int) -> pd.Series:
    """Compute rolling OLS regression coefficients for y ~ 1 + x1 + x2 + ...
    Arguments:
        x: DataFrame with columns 'y', 'x1', 'x2', ...
        window: Size of the rolling window
        coeff: Coefficient index to return (0=intercept, 1=x1, 2=x2)
    Returns:
        pd.Series with the desired rolling regression coefficient for each date
    """
    def _ols_coeffs(y, X) -> np.ndarray:
        """OLS regression: y ~ 1 + X
        Returns: array of [intercept, beta1, beta2, ..., mean squared residuals]
        """
        X = np.column_stack([np.ones(len(X)), X])
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            #betas, residuals = np.array([np.nan] * X.shape[1]), [np.nan]
            return np.array([np.nan] * (X.shape[1] + 1))
        else:
            try:
                betas, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                betas = np.linalg.pinv(X) @ y  # fallback to pseudo-inverse
                residuals = [np.sum((y - X @ betas)**2)]
            return np.concatenate([betas, [residuals[0]/len(residuals[0])]])
    # [betas[0]/((residuals[0]/len(X))**0.5)]])
    
    results = []
    for end in range(window, len(x) + 1):
        y = x.iloc[end - window : end, 0].values
        X = x.iloc[end - window : end, 1 :].values
        betas = _ols_coeffs(y, X)
        results.append(betas[coeff])
    if not results:
        return pd.Series([np.nan] * len(x), index=x.index)
    else:
        # pad the beginning with NaNs
        results = [np.nan] * (window - 1) + results
        return pd.Series(results, index=x.index)


def regression_residuals(x: pd.DataFrame) -> pd.Series:
    """Compute residuals from OLS regression of y ~ 1 + x1 + x2 + ...
    Arguments:
        x: DataFrame with columns 'y', 'x1', 'x2', ...
    Returns:
        pd.Series with the time-series of regression residuals
    """
    def _ols_residuals(y, X) -> np.ndarray:
        """OLS regression: y ~ 1 + X
        Returns: residuals of the regression or None
        """
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

#
# Panel Advanced Functions on Stock Characteristics
#
def characteristics_coalesce(*panels, replace: List = []) -> Panel:
    """Coalesce non-missing values from other Panels in order

    Arguments:
        panels: Panels to use for coalescing values
        replace: List of values, in addition to nan, considered to be missing
    Returns:
        Panel with the coalesced values
    """

    def replace_helper(x, replace: List) -> pd.Series:
        """Helper to replace NaN or listed values in the first column with values from the second column"""
        x[x.columns[0]] = x[x.columns[0]].fillna(x[x.columns[1]])
        # x.iloc[:, 0] = x.iloc[:, 0].fillna(x.iloc[:, 1].values)
        mask = x.iloc[:, 0].isin(replace)
        x.loc[mask, x.columns[0]] = x.loc[mask, x.columns[1]]
        return x.iloc[:, 0]

    if not is_list_like(replace):
        replace = [replace]
    out_panel = Panel()
    for panel in panels:
        out_panel = out_panel.apply(
            replace_helper, panel, how="outer", fill_value=np.nan, replace=replace
        )
    return out_panel


def characteristics_resample(characteristics: Panel, ffill: bool = True, month: List | int = []) -> Panel:
    """
    Resample a characteristics Panel to lower-frequency target dates (e.g., month-ends),
    with optional forward-filling within each sampling window.

    Use this when you need to **downsample** characteristics to specific calendar
    sampling points (controlled by `month`) of a lower-frequency set of target dates,
    and you optionally want to **carry forward** the most recently observed value for 
    each entity up to each sampled date if `ffill=True`.

    What this does
    --------------
    - Builds a calendar from the first to last available date.
    - Chooses target dates:
      - If `month` is empty, uses all months in the range.
      - If `month` is an int or list of ints (1–12), uses only those months.
    - For each target date and entity:
      - If `ffill=True`: uses the latest observed value on or before the target date
        **since the previous target date** (i.e., within the window).
      - If `ffill=False`: includes values **only** when an observation exists exactly
        on the target date.

    Parameters
    ----------
    characteristics : Panel
        A Panel with a 2-level index (date, stock_id) containing cross-sectional
        characteristics observations.

    ffill : bool, default True
        If True, forward-fill each entity’s characteristics to each sampled target date
        using the latest observation within the sampling window.
        If False, keep only observations that occur exactly on the sampled target date.

    month : list[int] | int | [], optional
        Target sampling months.
        - []: sample all months in the calendar range.
        - int or list[int] in 1..12: sample only those months (e.g., [3, 9] for Mar/Sep).

    Returns
    -------
    Panel
        A new Panel indexed by (sampled_date, stock_id) where each sampled date
        contains either:
        - the latest known value carried forward within the window (`ffill=True`), or
        - only exact-on-date observations (`ffill=False`).
    """
    assert characteristics.nlevels == 2, "characteristics must have two index levels"

    characteristics_dates = characteristics.dates
    prev_date = characteristics_dates[0]
    cal = Calendar(
        start_date=characteristics_dates[0], end_date=characteristics_dates[-1]
    )
    samples_df = []
    for next_date in cal.dates_range(cal.start_date, cal.end_date):
        if not month or cal.ismonth(next_date, month):
            for curr_date in cal.dates_range(prev_date, next_date):
                if curr_date in characteristics_dates:
                    if ffill or next_date == curr_date:
                        # stuff any observations after last date into samples_df, keep last later
                        characteristics_df = characteristics.frame.xs(
                            curr_date, level=0
                        ).reset_index()
                        characteristics_df[DATE_NAME] = next_date
                        characteristics_df["_date_"] = curr_date
                        samples_df.append(characteristics_df)
            prev_date = cal.offset(next_date, 1)

    # sort by STOCK_NAME, DATE_NAME and _date_ and drop duplicates, keep last
    samples_final = pd.concat(samples_df, axis=0)
    samples_final = samples_final.sort_values(by=[STOCK_NAME, DATE_NAME, "_date_"])
    samples_final = samples_final.drop_duplicates(subset=[STOCK_NAME, DATE_NAME], keep="last")
    samples_final = samples_final.set_index([DATE_NAME, STOCK_NAME]).drop(columns=["_date_"])
    samples_panel = Panel(samples_final)
    return samples_panel


#
# Panel Advanced Functions on Portfolio Weights
#
def portfolio_impute(port_weights: Panel, normalize: bool = True, drifted: bool = False) -> Panel:
    """Impute missing portfolio weights on missing dates by forward drifting previous weights based on
    stock price changes.
    Arguments:
        port_weights: Panel of portfolio weights.
        normalize: If True, re-normalize weights to be dollar-neutral after forward drifting.
        drifted: If True, output drifted weights every month for calculating turnover;
                 Else only fill in missing dates.
    Returns:
        Panel of portfolio weights with missing dates imputed by forward drifting based on stock price changes
    Notes:
        Side effect: Changes port_weights in place where missing dates are added.
    """
    # print('port_weights', port_weights.frame)  ###
    
    assert port_weights.nlevels == 2, "Portfolio weights must have two index levels"
    # should be ending dates of observed return, to align with dates of weights after drifting
    dates = dict(start_date=None, end_date=None)
    if CRSP_VERSION:
        retx = Panel().load("RETX", **dates)
    else:
        retx = Panel().load("ret_exc_lead1m", **dates).shift(1)
    portfolio_dates = port_weights.dates
    cal = Calendar(start_date=portfolio_dates[0], end_date=portfolio_dates[-1])
    all_dates = cal.dates_range(cal.start_date, cal.end_date)
    if len(all_dates) == len(portfolio_dates) and not drifted:
        return port_weights  # no missing dates to impute

    # pre-compute long and short notional on first date
    long_notional = port_weights.frame.xs(portfolio_dates[0], level=0)
    long_notional = long_notional[long_notional > 0].sum().abs().iloc[0]
    short_notional = port_weights.frame.xs(portfolio_dates[0], level=0)
    short_notional = short_notional[short_notional < 0].sum().abs().iloc[0]

    prev_weights = None
    drifted_weights = []
    for date in tqdm(all_dates, desc="portfolio_impute"):
        if (drifted or date not in portfolio_dates) and prev_weights is not None:
            # forward drift previous weights if any
            if retx is not None and date in retx.frame.index.get_level_values(0):
                # using retx returns to drift previous weights
                returns = retx.frame.xs(date, level=0).reindex(prev_weights.index, fill_value=0)
                curr_weights = (prev_weights.iloc[:, 0] * (1 + returns.iloc[:, 0])).to_frame()
                curr_weights.columns = prev_weights.columns
            if drifted:
                # store drifted weights in new frame if drifted requested
                new_weights = curr_weights.reset_index()
                new_weights[DATE_NAME] = date
                new_weights = new_weights.set_index([DATE_NAME, STOCK_NAME])
                drifted_weights.append(new_weights)

            # normalize weights if requested
            if normalize and long_notional > 0:
                curr_weights[curr_weights > 0] = (
                    long_notional
                    * curr_weights[curr_weights > 0]
                    / curr_weights[curr_weights > 0].abs().sum().iloc[0]
                )
            if normalize and short_notional > 0:
                curr_weights[curr_weights < 0] = (
                    short_notional
                    * curr_weights[curr_weights < 0]
                    / curr_weights[curr_weights < 0].abs().sum().iloc[0]
                )

            # add drifted weights to portfolio if date was missing
            if date not in portfolio_dates:
                curr_weights = curr_weights.dropna().reset_index()
                curr_weights[DATE_NAME] = date
                curr_weights = curr_weights.set_index([DATE_NAME, STOCK_NAME])
                port_weights._frame = pd.concat(
                    [port_weights.frame, curr_weights], axis=0
                )

        # update previous weights
        prev_weights = port_weights.frame.xs(date, level=0).copy()

    # finally, sort the portfolio weights by date and stock
    port_weights._frame = port_weights._frame.sort_index(level=[0, 1])
    if drifted:  # return all drifted weights if requested
        return Panel(pd.concat(drifted_weights, axis=0).sort_index(level=[0, 1]))
    else:  # only return imputed portfolio weights
        return port_weights


def portfolio_returns(port_weights: "Panel") -> "Panel":
    """Compute time series portfolio returns given portfolio weights
    Arguments:
        port_weights: Panel of portfolio weights
    Returns:
        Panel of portfolio returns, shifted by one date to align with end of holding period
    Note:
        Output is shifted by one date to align with ending dates of realized returns.
        If portfolio weights are missing on month-end dates, they will be imputed by drifting the prior month's.
    """
    # should be leading dates, to compute returns realized in the month ahead
    dates = dict(start_date=None, end_date=None)
    if CRSP_VERSION:
        stock_returns = Panel().load("EXCRET", **dates).shift(-1)
    else:
        stock_returns = Panel().load("ret_exc_lead1m", **dates)
    port_weights = portfolio_impute(port_weights, normalize=True)
    return (port_weights @ stock_returns).shift(1)


