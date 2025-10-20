import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from tqdm import tqdm
import json
from typing import List, Dict, Union, Set, Any, Tuple, Callable
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import warnings

DATA_PATH = Path('/home/terence/Downloads/scratch/2024/JKP/')
BENCH_PATH = DATA_PATH / 'factor-returns'
WORK_PATH = DATA_PATH / 'workspace'
MEDIA_PATH = DATA_PATH / 'media'

STOCK_NAME = 'permno'
DATE_NAME = 'eom'

###########################
#
# Data Cache library for storing intermediate DataFrames as parquet files
#
###########################
class DataCache:

    @staticmethod
    def load_cache() -> Dict[str, str]:
        """Load the data cache from the cache file."""
        cache_file = Path(WORK_PATH / 'cache.json')
        try:
            with open(cache_file, 'rb') as f:
                cache = json.load(f)
        except:
            cache = {"file_id": 0}
        return cache
    
    @staticmethod
    def dump_cache(cache: Dict[str, str]):
        """Dump the data cache to the cache file."""
        cache_file = Path(WORK_PATH / 'cache.json')
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=4)

    @staticmethod
    def write_frame(frame: pd.DataFrame, name: str = '', folder: str = WORK_PATH) -> str:
        """Write a dataframe to parquet file in the data cache
        Arguments:
            frame: DataFrame to write
            name: Optional name for the file, if not given, a new name will be generated
            folder: Folder to store the parquet files
        Returns:
            name: Name of the file (without extension)
        """
        if not name:
            cache = DataCache.load_cache()
            file_id = int(cache.get("file_id", 0)) + 1
            name = f"_{file_id}"
            cache["file_id"] = file_id
            DataCache.dump_cache(cache)
        frame.to_parquet(folder / f"{name}.parquet", index=True)
        return name

    @staticmethod
    def read_frame(name: str, folder: str = WORK_PATH) -> pd.DataFrame:
        """Read a dataframe from parquet file in the data cache
        Arguments:
            name: Name of the file (without extension)
            folder: Folder to read the parquet files from
        Returns:
            frame: DataFrame read from the file
        """
        return pd.read_parquet(folder / f"{name}.parquet")

    @staticmethod
    def reset(cache_path = WORK_PATH):
        """Resets data cache of PanelFrames"""
        for file in cache_path.glob("_*.parquet"):
            file.unlink()
        cache_file = Path(WORK_PATH / 'cache.json')
        if cache_file.exists():
            cache_file.unlink()

###########################
#
# Tools for Data Loader Agents
#
###########################

def csv_loader(filename: str | Path, 
               stock_name: str = STOCK_NAME, 
               date_name: str = DATE_NAME):
    """Load each column of CSV file (except stock and date indexes) to its own PanelFrame."""
    df_data = pd.read_csv(filename, sep='\t', header=0, low_memory=False)

    df_data['iid'] = df_data['iid'].astype(str)

    panel = PanelFrame('')
    for i, col in tqdm(enumerate(df_data.columns)):
        if col in [stock_name, date_name]:
            continue
        df = df_data[[date_name, stock_name, col]].dropna().convert_dtypes()
        df.set_index([date_name, stock_name], inplace=True)
        panel.name = col
#        df_combined = panel._append(df)
#        print(f"Saved {col}, shape now {df_combined.shape}, {df_combined.dtypes=}")
        #print(df_combined.head(5))

def load_benchmarks(characteristic: str = None, 
                    ret: str  = None, 
                    filename = 'USA.csv',
                    bench_path: Path = BENCH_PATH) -> pd.DataFrame:
    """Read benchmark returns from CSV file
    - 'ret_vw_cap': capped value-weighted return
    """
    df = pd.read_csv(bench_path / filename)
    if characteristic:
        df = df.loc[df['characteristic'] == characteristic, ['eom', ret]].set_index('eom').sort_index()
        df.index.name = DATE_NAME
        df.columns = [characteristic]
    else:
        if ret:
            # only keep date and ret columns and pivot to wide format
            df = df.pivot(index='eom', columns='characteristic', values=ret).sort_index()
            df.index.name = DATE_NAME
    return df

def load_variables(filename = 'variables.txt', 
                   data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Read names, types and descriptions from the variables.txt file"""

    names, types, descriptions = [], [], []
    with open(Path(data_path) / filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().lower().startswith('name '):
                continue  # skip header
            # Use regex to extract: name type length description
            match = re.match(r'^(\S+)\s+(\S+)\s+\S+\s+(.*)', line.strip())
            if match:
                name, typ, desc = match.group(1), match.group(2), match.group(3)
                names.append(name)
                types.append(typ)
                descriptions.append(desc)
    df = pd.DataFrame({'Type': types, 'Description': descriptions}, index=names)
    df.index.name = 'Name'
    return df


###########################
#
# Data Structures to support tools for Research Agents
#
###########################

class PanelFrame:
    """
    A PanelFrame is a wrapper around a pandas DataFrame with multi-index (date, stock),
    representing a panel data structure commonly used in finance and econometrics.
    It supports various operations such as arithmetic, logical, grouping, and advanced
    operations on the panel data.
    """            

    def __init__(self, name: str = '', start_date: str = '2020-01-01', end_date: str = '2024-12-31'):
        """Initialize a PanelFrame, optionally from a cached DataFrame file and date range.

        Arguments:
            name: Optionally load from named cached DataFrame file (without extension)
            start_date: Optional start date to filter the DataFrame (inclusive)
            end_date: Optional end date to filter the DataFrame (inclusive)
        """
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        if name:
            self.frame = DataCache.read_frame(name)
            if self.nlevels > 0:
                if self.start_date:
                    self.frame = self.frame[self.frame.index.get_level_values(0) >= self.start_date]
                if self.end_date:
                    self.frame = self.frame[self.frame.index.get_level_values(0) <= self.end_date]
        else:
            self.frame = None

    #
    # Primitive helpers
    #
    @property
    def nlevels(self) -> int:
        """Number of index levels of this PanelFrame
        Returns:
            -1 if empty, 0 if scalar, 1 if single index level (date), 2 if multi-index (date, stock)
        """
        if self.frame is None:
            return -1
        elif self.frame.index.nlevels == 1 and self.frame.index.name is None:
            return 0
        else:
            return self.frame.index.nlevels

    @property
    def values(self) -> np.ndarray:
        """Return the values of this PanelFrame as a numpy array.
        Returns:
            None if empty, scalar value if scalar, 1D array if single index level, 2D array if multi-index
        """
        if self.nlevels < 0:
            return None
        elif self.nlevels == 0:
            return self.frame.iloc[0, 0]
        else:
            return self.frame.iloc[:, 0].values
        
    def to_frame(self) -> pd.DataFrame:
        """Return the underlying DataFrame of this PanelFrame."""
        return self.frame   

    def copy(self, deep: bool = True) -> 'PanelFrame':
        """Return a copy of this PanelFrame.
        Arguments:
            deep: If True, also copy the underlying DataFrame, otherwise just copy the reference
        Returns:
            A new PanelFrame with the same name, date range, and optionally a copy of the DataFrame
        """
        new_panel = PanelFrame()
        new_panel.name = self.name
        new_panel.start_date = self.start_date
        new_panel.end_date = self.end_date
        if deep and self.frame is not None:
            new_panel.frame = self.frame.copy()
        else:
            new_panel.frame = None
        return new_panel
        
    def join_frame(self, other: 'PanelFrame', fillna: Any, how: str) -> pd.DataFrame:
        """Helper to join columns from another PanelFrame, and return as a DataFrame
        Arguments:
            other: Another PanelFrame to join with, or a scalar value to add as a column
            fillna: Value to fill missing values in the other PanelFrame
            how: Type of join to perform ('left', 'right', 'inner', 'outer')
        Returns:
            df: DataFrame with the joined data
        """
        df = self.frame.copy()
        if other is None:
            pass
        elif isinstance(other, PanelFrame):
            other_df = other.frame
            if df.index.nlevels != other_df.index.nlevels:
                raise ValueError("Cannot join PanelFrames with different index levels")
            df = df.join(other_df, how=how, lsuffix='_1', rsuffix='_2')
        else:
            df['other'] = other
        
        if fillna is not None and other is not None:
            df.iloc[:, 1] = df.iloc[:, 1].fillna(fillna)    # fillna only on the other column
        return df


    #
    # Primitive operations
    #
    def set_name(self, name: str) -> 'PanelFrame':
        """Set the name of this PanelFrame."""
        self.name = name
        return self
    
    def set_frame(self, frame: pd.DataFrame, append=True) -> 'PanelFrame':
        """Helper to set or append a DataFrame to this PanelFrame."""

        def _scalar_as_frame(frame: Any, col: str = '') -> pd.DataFrame:
            """Helper to convert a scalar to a DataFrame with one row and column, and no index name."""
            if not col and hasattr(frame, 'index'):
                col = frame.index[0]
            if hasattr(frame, 'values'):
                frame = frame.values
            frame = pd.DataFrame(frame, index=[col], columns=[col])
            frame.index.name = None
            return frame

        def _is_scalar(frame: pd.DataFrame) -> bool:
            """Helper to check if a DataFrame should be considered a scalar (1x1)"""
            return (frame.index.nlevels == 1 and   # only one index level
                    len(frame) == 1 and            # only one row
                    frame.index.name not in [DATE_NAME, STOCK_NAME]) # index name is not date or stock


        if _is_scalar(frame):
            self.frame = _scalar_as_frame(frame)
        else:
            if append and self.frame is not None:
                old_frame = self.frame
                frame.columns = old_frame.columns # force new column names match existing

                # drop duplicates based on index, keep the last occurrence
                self.frame = pd.concat([old_frame, frame], axis=0).sort_index(level=range(frame.index.nlevels))
                self.frame = self.frame[~self.frame.index.duplicated(keep='last')]
            else:
                self.frame = frame.sort_index(level=range(frame.index.nlevels))
            self.frame.index.names = [DATE_NAME, STOCK_NAME][:frame.index.nlevels] # ensure index names
        return self

    def set_dtype(self, dtype) -> 'PanelFrame':
        """Change the dtype of the values of this PanelFrame."""
        self.frame = self.frame.astype(dtype)
        return self

    def shift_dates(self, shift: int = 1) -> 'PanelFrame':
        """Shift the dates of this PanelFrame"""
        if self.nlevels <= 0:
            out_panel = self.copy(deep=True)
        else:
            out_panel = self.copy(deep=False)
            df = self.frame.reset_index(inplace=False)
            nlevels = df.index.nlevels

            # Create dictionary to map original dates to shifted dates
            unique_dates = sorted(df[DATE_NAME].unique())
            if shift > 0:
                zipped_dates = zip(unique_dates[:-shift], unique_dates[shift:])
            else:
                zipped_dates = zip(unique_dates[-shift:], unique_dates[:shift])
            date_map = {date_from: date_to for date_from, date_to in zipped_dates}

            # drop rows with dates that cannot be shifted
            df = df[df[DATE_NAME].isin(date_map)]

            # Replace dates using the mapping dictionary
            df[DATE_NAME] = df[DATE_NAME].replace(date_map)

            # Re-set the index and re-sort
            out_panel.frame = df.set_index([DATE_NAME, STOCK_NAME][:nlevels]).sort_index(level=list(range(nlevels)))
        return out_panel

    def persist(self, name: str = '') -> 'PanelFrame':
        """Set this PanelFrame to persist its data to cache file.
        Arguments:
            name: Optional name for the file, if not given, a new name will be generated
        Returns:
            self: This PanelFrame
        """
        name = DataCache.write_frame(frame=self.frame, name=name)
        self.name = name
        return self
    
    #
    # PanelFrame Binary Operators
    #
    def _operands(self, other: 'PanelFrame', fillna: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Internal helper to align another PanelFrame or scalar to this PanelFrame"""
        df = self.frame
        if isinstance(other, PanelFrame):
            df_other = other.frame.reindex(df.index).fillna(fillna).values
        else:
            df_other = other
        return df, df_other

    def __add__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Add values of this PanelFrame with other"""
        df, df_other = self._operands(other, 0)
        return PanelFrame().set_frame(df + df_other)

    def __radd__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Add values of this PanelFrame with other"""
        df, df_other = self._operands(other, 0)
        return PanelFrame().set_frame(df_other + df)
    
    def __sub__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Subtract values of this PanelFrame with other"""
        df, df_other = self._operands(other, 0)
        return PanelFrame().set_frame(df - df_other)

    def __rsub__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Subtract values of this PanelFrame with other"""
        df, df_other = self._operands(other, 0)
        return PanelFrame().set_frame(df_other - df)
    
    def __mul__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Multiply values of this PanelFrame with other"""
        df, df_other = self._operands(other, 0)
        return PanelFrame().set_frame(df * df_other)
    
    def __rmul__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Multiply values of this PanelFrame with other"""
        df, df_other = self._operands(other, 0)
        return PanelFrame().set_frame(df_other * df)
    
    def __truediv__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Divide values of this PanelFrame with other"""
        df, df_other = self._operands(other, 0)
        return PanelFrame().set_frame(df / df_other)
    
    def __rtruediv__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Divide values of this PanelFrame with other"""
        df, df_other = self._operands(other, 0)
        return PanelFrame().set_frame(df_other / df)

    def __eq__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Check equality of values of this PanelFrame with other"""
        df, df_other = self._operands(other, np.nan)
        return PanelFrame().set_frame((df == df_other).astype(bool))
    
    def __ge__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Check equality of values of this PanelFrame with other"""
        df, df_other = self._operands(other, np.nan)
        return PanelFrame().set_frame((df >= df_other).astype(bool))

    def __gt__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Check equality of values of this PanelFrame with other"""
        df, df_other = self._operands(other, np.nan)
        return PanelFrame().set_frame((df > df_other).astype(bool))

    def __le__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Check equality of values of this PanelFrame with other"""
        df, df_other = self._operands(other, np.nan)
        return PanelFrame().set_frame((df <= df_other).astype(bool))

    def __lt__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Check equality of values of this PanelFrame with other"""
        df, df_other = self._operands(other, np.nan)
        return PanelFrame().set_frame((df < df_other).astype(bool))

    def __ne__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Check inequality of values of this PanelFrame with other"""
        df, df_other = self._operands(other, np.nan)
        return PanelFrame().set_frame((df != df_other).astype(bool))

    def __or__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Logical or of values of this PanelFrame with other"""
        df, df_other = self._operands(other, None)
        return PanelFrame().set_frame((df.astype(bool) | df_other.astype(bool)))

    def __and__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Logical or of values of this PanelFrame with other"""
        df, df_other = self._operands(other, None)
        return PanelFrame().set_frame((df.astype(bool) & df_other.astype(bool)))

    #
    # PanelFrame Unary Operators
    #
    def __neg__(self) -> 'PanelFrame':
        """Negate the values of this PanelFrame."""
        return PanelFrame().set_frame(-self.frame)
    
    def __not__(self) -> 'PanelFrame':
        """Logical NOT of the values of this PanelFrame."""
        return PanelFrame().set_frame(~self.frame.astype(bool))

    def log(self) -> 'PanelFrame':
        """Logarithm of the values of this PanelFrame."""
        return PanelFrame().set_frame(self.frame.apply(np.log))

    def exp(self) -> 'PanelFrame':
        """Exponentiate the values of this PanelFrame."""
        return PanelFrame().set_frame(self.frame.apply(np.exp))

    #
    # PanelFrame Utilities
    #
    def plot(self, other_panel: 'PanelFrame' = None, **kwargs):
        """Plot the values of this PanelFrame.
        Arguments:
            other_panel: Optional other PanelFrame to plot on the same axes
            kwargs: keyword arguments to pass to pandas.DataFrame.plot()
        """
        df = self.frame.copy()
        if other_panel is not None:
            df = df.join(other_panel.frame, how='outer', rsuffix='_2')
            if 'x' not in kwargs:
                kwargs['x'] = df.columns[0]
            if 'y' not in kwargs:
                kwargs['y'] = df.columns[1]
        df.plot(**kwargs)

    #
    # PanelFrame Group and Advanced Operations
    #
    def apply(self, func: Callable, reference: 'PanelFrame' = None, fillna=0, how='left', **kwargs) -> 'PanelFrame':
        """Apply a function to each date group of PanelFrame, optionally based on values reference PanelFrame.
        Arguments:
            func: function to apply to each date group, must accept a DataFrame and return a Series
            reference: optional PanelFrame to join with before applying the function
            fillna: value to fill missing values in the reference PanelFrame
            how: type of join to perform with the reference PanelFrame ('left', 'right', 'inner', 'outer')
            kwargs: additional keyword arguments to pass to the function
        Returns:
            PanelFrame with the same index as this PanelFrame, with values computed by the function
        """
        df = self.join_frame(reference, fillna=fillna, how=how)
        cols = df.columns
        #print(df.index.nlevels, cols, df.shape)

        # Input has single index level
        if df.index.nlevels == 1:
            df = func(df, **kwargs)  # apply function directly 

        # Input has two index levels
        else:
            # apply function to each group by the first index level (date)
            df = df.groupby(level=0).apply(func, **kwargs)

            # flatten extra index levels   
            while df.index.nlevels > 2:
                df = df.reset_index(level=0, drop=True)

        # if the result is a Series, convert to DataFrame
        if hasattr(df, 'to_frame'):
            df = df.to_frame()
        df.columns = cols[:len(df.columns)]
        return PanelFrame().set_frame(pd.DataFrame(df).iloc[:, [0]])

    def __matmul__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Compute the dot product of two PanelFrames, by first index level date group."""
        def dot(x):
            """Dot product of two columns"""
            return (x.iloc[:, 0] * x.iloc[:, -1]).sum()
        return self.apply(dot, other, fillna=0)

#
# Common functions to be used with PanelFrame.apply()
#

def weighted_average(x):
    """
    Compute the weighted average of the first column, weighted by the last column.
    Arguments:
        x: DataFrame with at least two columns, first column is the data to be averaged,
           last column is the weight for each row
    Returns:
        float: Weighted average of the first column
    Usage:
        panel_frame.apply(weighted_average, weights or 1, fillna=0)
    """
    return (x.iloc[:, 0] * x.iloc[:, -1]).sum() / x.iloc[:, -1].sum()


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
        panel_frame.apply(winsorize, indicator or True, fillna=False, lower=lower, upper=upper)     
    """
    lower, upper = x.loc[x.iloc[:,-1].astype(bool), x.columns[0]].quantile([lower, upper]).values
    return x.iloc[:, 0].clip(lower=lower, upper=upper)


def quantiles(x, num) -> pd.Series:
    """
    Assign quantiles (1 to num) to the first column based on the quantiles of the true rows in the last column.

    Arguments:
        x: DataFrame with at least two columns, first column is the data to be quantiled,
           last column is a boolean indicator for which rows to consider for quantiling
        num: Number of quantiles to create
    Returns:
        pd.Series with quantile assignments (1 to num) for each row in the first column
    Usage:
        panel_frame.apply(quantiles, other_frame or True, fillna=False, num=num)     
    """
    edges = pd.qcut(x.loc[x.iloc[:,-1].astype(bool), x.columns[0]], q=num, labels=False, retbins=True)[1]
    bins = pd.cut(x.iloc[:,0], bins=edges, labels=False, include_lowest =True)
    return bins.fillna(0).astype(int) + 1


def spread_portfolios(x) -> pd.Series:
    """
    Compute long portfolio weights of the highest and short portfolio weights of the lowest quantile
    of the first column, weighted by the second column if given, otherwise equal weight.
    Other quantiles are assigned zero weight.

    Arguments:
        x: DataFrame with at least two columns, first column is the quantile assignment (-1, 0, 1),
           second column is the relative weight for each row
    Returns:
        pd.Series with the spread portfolio weights for each row in the first column
    Usage:
        panel_frame.apply(spread_portfolios, weights or 1, fillna=0)
    """
    low_quantile = x.iloc[:, 0].min()
    high_quantile = x.iloc[:, 0].max()
    other_quantile = ~x.iloc[:, 0].isin([low_quantile, high_quantile])
    x.iloc[:, 0] = x.iloc[:, 0].replace({low_quantile: -1, high_quantile: 1})
    x.iloc[other_quantile, 0] = 0
    x['_total_weight'] = x.groupby(x.columns[0])[x.columns[1]].transform('sum') # normalize weights
    return x.iloc[:, 0].mul(x.iloc[:, 1]).div(x['_total_weight']).rename(x.columns[0])

class PortfolioEvaluation:
    """Evaluate the performance of a portfolio PanelFrame"""

    def __init__(self, portfolio: PanelFrame):
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


class FactorEvaluation:
    """Evaluate the performance of a factor returns PanelFrame"""
    def __init__(self, factor: PanelFrame, annualization: int = 12):
        if factor.nlevels != 1:
            raise ValueError("FactorEvaluation requires a PanelFrame with single index level (date)")
        self.factor = factor
        self.annualization = annualization

    def volatility(self) -> float:
        """Compute the annualized volatility of the factor returns."""
        return self.factor.frame.std().values[0] * np.sqrt(self.annualization)
    
    def annualized_return(self) -> float:
        """Compute the arithmetic annualized return of the factor returns."""
        avg_ret = self.factor.frame.mean().values[0]
        return avg_ret * self.annualization
    
    def sharpe_ratio(self) -> float:
        """Compute the annualized Sharpe ratio of the factor returns."""
        vol = self.volatility()
        if vol == 0:
            return 0.0
        return self.annualized_return() / vol
    
    def max_drawdown(self) -> float:
        """Compute the maximum drawdown of the factor returns."""
        cum_ret = (1 + self.factor.frame).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        return drawdown.min().values[0]

    def summary(self) -> Dict[str, float]:
        """Compute a summary of the factor performance metrics."""
        return {
            'Annualized Return': self.annualized_return(),
            'Volatility': self.volatility(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Max Drawdown': self.max_drawdown()
        }

#
# Running code on the Data Cache 
#
import subprocess, sys, os#
def run_code_in_subprocess(code_str):
    env = os.environ.copy()
    # prepend your project root to PYTHONPATH
    env["PYTHONPATH"] = "/home/terence/Dropbox/github/thesis/src:" + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, "-c", code_str],
        capture_output=True,
        text=True,
        env=env
    )
    print(f"Subprocess exited with code {proc.returncode}")
    return proc.stdout, proc.stderr, proc.returncode    


if __name__ == "__main__":

    print(str(datetime.now()))

    # Unit Test 6: pipeline
    ret = 'ret_vw_cap'
    factor = 'ret_12_1'
    dates = dict(start_date='1975-01-01', end_date='2024-12-31')
    dates = dict(start_date='2020-01-01', end_date='2024-12-31')

    factor_pf = PanelFrame(factor, **dates)

    size_pf = PanelFrame('me', **dates)

    nyse_pf = PanelFrame('crsp_exchcd', **dates).apply(pd.DataFrame.isin, values=[1, '1'])

    decile_pf = size_pf.apply(quantiles, nyse_pf, num=10)

    quantiles_pf = factor_pf.apply(quantiles, decile_pf > 1, num=3)

    vwcap_pf = PanelFrame('me', **dates).apply(winsorize, nyse_pf, lower=0, upper=0.80)

    spreads_pf = quantiles_pf.apply(spread_portfolios, vwcap_pf)

    lead1m_pf = PanelFrame('ret_exc_lead1m', **dates)
    facret_pf = (spreads_pf @ lead1m_pf).shift_dates(shift=1)

    bench = PanelFrame(factor + '_' + ret, **dates)
    facret_pf.plot(bench, kind='scatter')

    print(facret_pf)
    print(str(datetime.now()))

    panel_id = 'me'
    other_panel_id = 'ret_exc_lead1m'
    code = f"""
import json
from qrafti import PanelFrame
dates = dict(start_date='2020-01-01', end_date='2024-12-31')
p1, p2 = PanelFrame('{panel_id}', **dates), PanelFrame('{other_panel_id}', **dates)
p3 = (p1 @ p2).persist()
print(json.dumps({{'result_panel_id': p3.name}}))
"""
    stdout, stderr, exit_code = run_code_in_subprocess(code)
    print(exit_code)
    print(stdout)
    raise Exception

    code = """
# Please run this following code:
from qrafti import PanelFrame, MEDIA_PATH
import matplotlib.pyplot as plt
ret = 'ret_vw_cap'
factor = 'ret_12_1'
bench = PanelFrame(factor + '_' + ret).to_frame()
bench.cumsum().plot()
savefig = MEDIA_PATH / f"{factor}_{ret}.png"
plt.savefig(savefig)
print(f"[Image File](file:///{str(savefig)})")
"""
#

### Load variables
#    df_vars = load_variables()
#    print(str(datetime.now()))
#    DataCache.reset('*')
#    for filename in tqdm(sorted((DATA_PATH / 'USA').glob('19[8765432]*.txt.gz'), reverse=True)):
#        print(f"Loading {filename}")
#        csv_loader(filename)
#    print(str(datetime.now()))

    ret = 'ret_vw_cap'

### Load benchmarks
    # benchmarks_df = load_benchmarks(ret=ret)
    # for bench in benchmarks_df.columns:
    #     df = benchmarks_df[[bench]].dropna()
    #     df.columns = [bench + '_' + ret]
    #     PanelFrame(bench+ '_' + ret).to_cache(df, append=False)
    # print(str(datetime.now()))

### Unit Test 0: cap weighted returns
#     stock_caps = PanelFrame('me', start_date='1995-01-31', end_date='2024-12-31')
#     stock_rets = PanelFrame('ret_exc_lead1m', start_date='1995-01-31', end_date='2024-12-31')
#     mkt = stock_rets.weighted_average(stock_caps)
#     (1+mkt).log().cumsum().exp().to_frame().plot()
#     plt.show()

#     code = """
# # Please run this following code:
# from qrafti import PanelFrame, MEDIA_PATH
# import matplotlib.pyplot as plt
# stock_caps = PanelFrame('me', start_date='1995-01-31', end_date='2024-12-31')
# stock_rets = PanelFrame('ret_exc_lead1m', start_date='1995-01-31', end_date='2024-12-31')
# mkt = stock_rets.weighted_average(stock_caps)
# (1+mkt).log().cumsum().exp().to_frame().plot()
# savefig = MEDIA_PATH / f"capwtd_mkt.png"
# plt.savefig(savefig)
# print(f"[Image File](file:///{str(savefig)})")
# """
#     stdout, stderr, exit_code = run_code_in_subprocess(code)

    # Unit Test 1: join
    p1 = PanelFrame('prc', start_date='2020-01-31', end_date='2020-03-31')
    df1 = p1.to_frame()
    print(p1)
    print(p1.join_frame(PanelFrame('ret'), how='inner', fillna=0).head(10))

    # Unit Test 2: winsorize
    p2 = p1.winsorize()
    df2 = p2.to_frame()
    print(p2)


    # Unit Test 5: equality

    p3 = PanelFrame('crsp_exchcd', start_date='2020-01-31', end_date='2020-03-31').isin([1, '1'])
    df3 = p3.to_frame()
    (p3 == 1).to_frame().head(10)
    
    p4 = p1.winsorize(indicator=p3)
    df4 = p4.to_frame()

#    raise Exception
    #"""
    #"""
    print(DataCache.file_id, len(DataCache.cache_memory))

#    raise Exception

    factor_id = p7.name
#    factor_id = factor + '_' + ret
    kwargs = {'kind': 'scatter'}
    kwargs = {}
    args = (factor + '_' + ret,)
    args = tuple()
    code_str = f"""
from qrafti import PanelFrame, MEDIA_PATH
import matplotlib.pyplot as plt
p1 = PanelFrame('{factor_id}')
if {len(args)} > 0:
    args = (PanelFrame(arg) for arg in {args})
    fig = p1.plot(*args, **{kwargs})
    print(args[0].name)
else:
    fig = p1.plot(**{kwargs})
savefig = MEDIA_PATH / f"plot_{factor_id}.png"
plt.savefig(savefig)
print(f"[Image File](file:///{{savefig}})")
"""
    stdout, stderr, exit_code = run_code_in_subprocess(code_str)
    print(exit_code)
    print(stderr)
    print(stdout)
