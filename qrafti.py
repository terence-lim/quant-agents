# qrafti.py  (c) Terence Lim

from utils import DataCache, markdown_to_pdf, Calendar, MEDIA, STOCK_NAME, DATE_NAME
from portfolio import PortfolioEvaluation
from rag import RAG

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from tqdm import tqdm
import json
from typing import List, Dict, Union, Set, Any, Tuple, Callable
from datetime import datetime
import time
import pandas as pd
from pandas.api.types import is_list_like, is_scalar, is_numeric_dtype, is_float_dtype
import matplotlib.pyplot as plt
import os
import logging
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
logging.disable(logging.DEBUG)
#pd.set_option("future.no_silent_downcasting", True)  # for fillna behavior


#
# These are temporarily here, should be in utils.py...
#
DATES = dict(start_date="2020-01-01", end_date="2024-12-31")
#DATES = dict(start_date="2001-01-01", end_date="2024-12-31")
#DATES = dict(start_date="1960-01-01", end_date="2024-12-31")

CRSP_VERSION = True

if CRSP_VERSION:
    RAG_PATH = Path("/home/terence/Downloads/scratch/2024/JKP/CRSP_RAG")
else:
    RAG_PATH = Path("/home/terence/Downloads/scratch/2024/JKP/JKP_RAG")


###########################
#
# Data Structures to support tools for Research Agents
#
###########################

class Panel:
    """
    A Panel is a wrapper around a pandas DataFrame with multi-index (date, stock),
    representing a panel data structure commonly used in finance and econometrics.
    It supports various operations such as arithmetic, logical, grouping, and advanced
    operations on the panel data.
    """

    def __init__(self, data: Union[int, float, pd.DataFrame, pd.Series, "Panel"] = None, name: str = ""):
        """Initialize a Panel, optionally from a number, DataFrame or another Panel.

        Arguments:
            data: Can be a scalar value, DataFrame or a Panel
        """
        self.name = name
        if isinstance(data, (int, float, bool)):
            self._frame = pd.DataFrame(data, index=[""], columns=[""])
            self._frame.index.name = None
        elif isinstance(data, pd.DataFrame):
            assert all(s in [DATE_NAME, STOCK_NAME] for s in data.index.names)
            self._frame = data            
        elif isinstance(data, pd.Series):
            assert all(s in [DATE_NAME, STOCK_NAME] for s in data.index.names)
            self._frame = data.to_frame()
        elif data is None:
            self._frame = None
        elif isinstance(data, Panel):
            self._frame = None if data.nlevels < 0 else data._frame.copy()
        else:
            raise ValueError("Data must be a scalar value or a pandas DataFrame")
        if self.nlevels > 0:  # Ensure index order and unique
            if self.nlevels > 1:
                self._frame = self._frame.reorder_levels([DATE_NAME, STOCK_NAME])
            self._frame = self._frame.sort_index(level=range(self._frame.index.nlevels))
            self._frame = self._frame[~self._frame.index.duplicated(keep="last")]
    #
    # Property accessors
    #
    @property
    def frame(self) -> pd.DataFrame:
        """Return the underlying DataFrame or scalar value of this Panel."""
        if self.nlevels == 0:
            return self._frame.iloc[0, 0]
        return self._frame

    @property
    def nlevels(self) -> int:
        """Number of index levels of this Panel
        Returns:
            -1 if empty, 0 if scalar, 1 if single index level (date), 2 if multi-index (date, stock)
        """
        if self._frame is None:
            return -1
        elif (self._frame.index.nlevels == 1 and 
              self._frame.index.name not in [DATE_NAME, STOCK_NAME]):
            return 0
        else:
            return self._frame.index.nlevels

    @property
    def values(self) -> np.ndarray:
        """Return the values of this Panel as a numpy array.
        Returns:
            [] if empty, 1D array otherwise
        """
        return [] if self.nlevels < 0 else self.frame.iloc[:, 0].values

    @property
    def dates(self) -> List[str]:
        """Return the list of unique dates in this Panel."""
        return [] if self.nlevels <= 0 else sorted(self.frame.index.get_level_values(0).unique())

    def __len__(self) -> int:
        """Return the number of data items in this Panel."""
        return 0 if self._frame is None else len(self._frame)

    def as_payload(self) -> dict:
        """Returns the persisted name of this Panel, and number of index levels and rows as a dict"""
        return {"results_panel_id": self.name,
                "nlevels": self.nlevels,
                "rows": len(self)}

    #
    # Primitive operations
    #
    def __getitem__(self, key: Union[str, int, Tuple, slice]) -> pd.Series | float | int | None:
        """Get item(s) from this Panel using indexing.
        Arguments:
            key: Can be a date string, integer index, or tuple of (date, stock)
        Returns:
            The value(s) at the specified index, or None if not found
        """
        # Handle string range: obj['a':'c']
        # if isinstance(key, slice):
        #     start, stop = key.start, key.stop
        #     keys = [k for k in self.data if (start is None or k >= start)
        #                                    and (stop  is None or k <= stop)]
        #    return {k: self.data[k] for k in keys}
        try:        
            if isinstance(key, slice):
                start, stop = key.start, key.stop
                if isinstance(start, int) or isinstance(stop, int):
                    keys = [k for k in self._frame.index.get_level_values(STOCK_NAME)
                            if (start is None or k >= start) and (stop is None or k <= stop)]
                    return self._frame.loc[self._frame.index.isin(keys, level=STOCK_NAME)].iloc[:, 0]
                else:
                    if start is not None:
                        start = pd.to_datetime(start)
                    if stop is not None:
                        stop = pd.to_datetime(stop)
                    keys = [k for k in self._frame.index.get_level_values(DATE_NAME).unique()
                            if (start is None or k >= start) and (stop is None or k <= stop)]
                    return self._frame.loc[self._frame.index.isin(keys, level=DATE_NAME)].iloc[:, 0]
            elif self.nlevels == 1:
                return self.frame.loc[key].iloc[0, 0]
            else:  # nlevels == 2
                if isinstance(key, tuple) and len(key) == 2:  # (date, stock)
                    return self.frame.loc[key].iloc[0, 0]
                elif isinstance(key, int):  # return all dates for that stock
                    return self.frame.xs(key, level=STOCK_NAME).iloc[:, 0]
                else:   # return all stocks for that date
                    return self.frame.xs(key, level=DATE_NAME).iloc[:, 0]
        except Exception:
            return None

    def copy(self) -> "Panel":
        """Return a copy of this Panel.
        Arguments:
            deep: If True, also copy the underlying DataFrame, otherwise just copy the metadata
        Returns:
            A new Panel with the same name, date range, and optionally a copy of the DataFrame
        """
        new_panel = Panel()
        new_panel.name = self.name
        new_panel._frame = None if self.nlevels < 0 else self._frame.copy()
        return new_panel
    
    def drop(self, key: str | int) -> 'Panel':
        """Return a new Panel with the specified index key dropped.
        Arguments:
            key: The date string, integer index, or tuple of (date, stock) to drop
        Returns:
            A new Panel with rows of the the specified index key removed
        """
        if self.nlevels < 0:
            return self.copy()
        new_panel = self.copy()
        try:
            if self.nlevels == 1:
                new_panel._frame = new_panel._frame.drop(index=key)
            else:  # nlevels == 2
                if isinstance(key, tuple) and len(key) == 2:  # (date, stock)
                    new_panel._frame = new_panel._frame.drop(index=pd.Index([key], names=[DATE_NAME, STOCK_NAME]))
                elif isinstance(key, int):  # drop all dates for that stock
                    new_panel._frame = new_panel._frame.drop(index=key, level=STOCK_NAME)
                else:   # drop all stocks for that date
                    new_panel._frame = new_panel._frame.drop(index=key, level=DATE_NAME)
        except Exception:
            pass  # key not found, return unchanged copy
        return new_panel

    def ones_like(self) -> "Panel":
        """Sets values of this Panel equal to 1"""
        if self.nlevels < 0:
            return Panel(name=self.name)
        elif self.nlevels == 0:
            return Panel(1, name=self.name)
        else:
            frame = pd.DataFrame(1, index=self._frame.index, columns=self._frame.columns)
            return Panel(frame, name = self.name)

    def load(self, name: str, start_date: str = None, end_date: str = None) -> "Panel":
        """Load a cached DataFrame file into this Panel.
        Arguments:
            name: Name of the cached DataFrame file (without extension)
            start_date: Optional start date to filter the DataFrame (inclusive)
            end_date: Optional end date to filter the DataFrame (inclusive)
        Returns:
            self: This Panel with loaded data
        """
        self.name = name
        self._frame = DataCache.read_frame(name)  # may be None
        if self.nlevels > 0 and DATE_NAME in self._frame.index.names:
            if start_date:
                start_date = pd.to_datetime(start_date)
                self._frame = self._frame[
                    self._frame.index.get_level_values(DATE_NAME) >= start_date                        
                ]
            if end_date:
                end_date = pd.to_datetime(end_date)
                self._frame = self._frame[
                    self._frame.index.get_level_values(DATE_NAME) <= end_date
                ]
        return self
    
    def append(self, other: "Panel") -> "Panel":
        """Append another Panel to this Panel.
        Arguments:
            other: Another Panel to append
        Returns:
            self: This Panel with appended data
        """
        if self.frame is None:
            self.name = other.name
            self._frame = other._frame
        elif other.frame is not None:
            assert self.nlevels > 0 and other.nlevels > 0, "Cannot append scalar or empty Panels"
            assert self.frame.columns.equals(other.frame.columns), "Cannot append Panels with different columns"
            self._frame = pd.concat([self._frame, other._frame], axis=0)
            self._frame = self._frame.sort_index(level=range(self._frame.index.nlevels))
            self._frame = self._frame[~self._frame.index.duplicated(keep="last")]
        return self

    def save(self, name: str = "") -> "Panel":
        """Set this Panel to persist its data to cache file.
        Arguments:
            name: Optional name for the file, if not given, a new name will be generated
        Returns:
            self: This Panel
        """
        if name and self.nlevels >= 0:
            self._frame.columns = [name]
            if self.nlevels == 0:
                self._frame.index = [name]
        name = DataCache.write_frame(frame=self._frame, name=name)
        self.name = name
        return self


    #
    # Primitive helpers
    #
    def join_frame(self, other: "Panel", fill_value: Any, how: str, require_dates: bool) -> pd.DataFrame:
        """Helper to join columns from another Panel, and return as a DataFrame
        Arguments:
            other: Another Panel to join with, or a scalar value to add as a column
            fill_value: Value to fill missing values in the other Panel
            how: Type of join to perform ('left', 'right', 'inner', 'outer')
            require_dates: if True, then only include first-level dates that exist in both Panels
        Returns:
            df: DataFrame with the joined data
        """
        assert (self.nlevels != 0) or (isinstance(other, Panel) and other.nlevels > 0)
        if self.nlevels > 0:
            # self is a multi-index Panel:
            df = self.frame.copy()
        else:
            # self is a scalar Panel or None: create DataFrame with same index as other
            data = (
                fill_value if self._frame is None else self.frame
            )  # value for self Panel
            df = pd.DataFrame(index=other.frame.index, data=data, columns=["self"])

        if isinstance(other, Panel):  # other is a Panel
            if other.nlevels > 0:
                # assert df.index.nlevels == other_df.index.nlevels
                
                # other is also a multi-index Panel: join on index levels
                other_df = other.frame

                # keeping only common index level 0
                if require_dates and other_df.index.nlevels > 1 and df.index.nlevels > 1: 
                    common = df.index.remove_unused_levels().levels[0].intersection(
                        other_df.index.remove_unused_levels().levels[0])
                    df = df[df.index.get_level_values(0).isin(common)]
                    other_df = other_df[other_df.index.get_level_values(0).isin(common)] 

                df = df.join(other_df, how=how, rsuffix="_").fillna(fill_value)
            elif other.nlevels == 0:
                # other is a scalar Panel: add as a column with same scalar value
                df["other"] = other.frame
            else:
                # other is None: add as column with fill-value
                df["other"] = fill_value
        elif other is None:
            df["other"] = fill_value
        else:
            df["other"] = other
            # if is_float_dtype(df.dtypes.iloc[1]):
            #    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        if df.index.nlevels > 1:
            df.index = df.index.remove_unused_levels()
        return df


    #
    # Panel Binary Operators
    #
    def _operands(self, other: "Panel", fill_value: Any, how: str) -> Tuple[pd.Series, pd.Series]:
        """Internal helper to align another Panel or scalar to this Panel
        Notes: If other is a Panel, perform an outer join, and fill missing values with fill_value
        """
        if isinstance(other, Panel):
            assert (self.nlevels >= 0) and (other.nlevels >= 0), "Cannot operate on empty Panels"
            df = self.join_frame(other, fill_value=fill_value, how=how, require_dates=False)
            df_other = df.iloc[:, 1]
            df = df.iloc[:, 0]
        else:  # other is a scalar
            assert isinstance(other, (int, float, bool)), "Other operand must be a Panel or scalar" 
            if self.nlevels <= 0:   # self is a scalar or empty
                df = fill_value if self._frame is None else self.frame
            else:  # self is a multi-index Panel
                df = self._frame.iloc[:, 0]
            df_other = other
        return df, df_other

    def __add__(self, other: "Panel") -> "Panel":
        """Add values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=0, how="outer")
        return Panel(df + df_other)

    def __radd__(self, other: "Panel") -> "Panel":
        """Add values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=0, how="outer")
        return Panel(df_other + df)

    def __sub__(self, other: "Panel") -> "Panel":
        """Subtract values of other Panel from this Panel"""
        df, df_other = self._operands(other, fill_value=0, how="outer")
        return Panel(df - df_other)

    def __rsub__(self, other: "Panel") -> "Panel":
        """Subtract values of this Panel from other"""
        df, df_other = self._operands(other, fill_value=0, how="outer")
        return Panel(df_other - df)

    def __mul__(self, other: "Panel") -> "Panel":
        """Multiply values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=1, how="inner")
        return Panel(df * df_other)

    def __rmul__(self, other: "Panel") -> "Panel":
        """Multiply values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=1, how="inner")
        return Panel(df_other * df)

    def __truediv__(self, other: "Panel") -> "Panel":
        """Divide values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=1, how="inner")
        return Panel(df / df_other)

    def __rtruediv__(self, other: "Panel") -> "Panel":
        """Divide values of other Panel by this Panel"""
        df, df_other = self._operands(other, fill_value=1, how="inner")
        return Panel(df_other / df)

    def __eq__(self, other: "Panel") -> "Panel":
        """Check equality of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how="inner")
        return Panel((df == df_other).astype(bool))

    def __ge__(self, other: "Panel") -> "Panel":
        """Check equality of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how="inner")
        return Panel((df >= df_other).astype(bool))

    def __gt__(self, other: "Panel") -> "Panel":
        """Check equality of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how="inner")
        return Panel((df > df_other).astype(bool))

    def __le__(self, other: "Panel") -> "Panel":
        """Check equality of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how="inner")
        return Panel((df <= df_other).astype(bool))

    def __lt__(self, other: "Panel") -> "Panel":
        """Check equality of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how="inner")
        return Panel((df < df_other).astype(bool))

    def __ne__(self, other: "Panel") -> "Panel":
        """Check inequality of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how="inner")
        return Panel((df != df_other).astype(bool))

    def __or__(self, other: "Panel") -> "Panel":
        """Logical or of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=0, how="outer")
        return Panel((df.astype(bool) | df_other.astype(bool)))

    def __and__(self, other: "Panel") -> "Panel":
        """Logical or of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=1, how="inner")
        return Panel((df.astype(bool) & df_other.astype(bool)))

    def __pow__(self, other: "Panel") -> "Panel":
        """Raise values of this Panel by the other"""
        df, df_other = self._operands(other, fill_value=1, how="inner")
        return Panel(df**df_other)

    def __rpow__(self, other: "Panel") -> "Panel":
        """Raise values of other Panel by this Panel"""
        df, df_other = self._operands(other, fill_value=1, how="inner")
        return Panel(df_other**df)

    def pow(self, other: int | float) -> "Panel":
        """Raise values of this Panel by a value"""
        return Panel(np.pow(self.frame, other))

    #
    # Panel Unary Operators
    #
    def __neg__(self) -> "Panel":
        """Negate the values of this Panel."""
        return Panel(-self.frame)

    def __invert__(self) -> "Panel":
        """Boolean negation of the values of this Panel."""
        return Panel(~self.frame.astype(bool))

    def log(self) -> "Panel":
        """Logarithm of the values of this Panel."""
        return Panel(self.frame.apply(np.log))

    def exp(self) -> "Panel":
        """Exponentiate the values of this Panel."""
        return Panel(self.frame.apply(np.exp))

    def log1p(self) -> "Panel":
        """Logarithm of 1 plus the values of this Panel."""
        return Panel(self.frame.apply(np.log1p))

    def expm1(self) -> "Panel":
        """Exponentiate the values of this Panel minus 1."""
        return Panel(self.frame.apply(np.expm1))

    def abs(self) -> "Panel":
        """Absolute values of this Panel."""
        return Panel(self.frame.apply(np.abs))

    def int(self) -> "Panel":
        """Convert to integer type"""
        return Panel(self.frame.astype(int))

    #
    #
    # Panel Functions to Apply by Group
    #
    def apply(self, func: Callable, reference: "Panel" = None, fill_value=0, how="left", **kwargs) -> "Panel":
        """Apply a function to each date group of Panel, optionally based on values reference Panel.
        Arguments:
            func: function to apply to each date group, must accept a DataFrame and return a Series
            reference: optional Panel to join with before applying the function
            fill_value: value to fill missing values in the reference Panel
            how: type of join to perform with the reference Panel, default is 'left'
            kwargs: additional keyword arguments to pass to the function
        Returns:
            Panel with the same index as this Panel, with values computed by the function
        """
        # assert self.nlevels > 0, "Cannot apply function to scalar empty Panel"
        df = self.join_frame(reference, fill_value=fill_value, how=how, require_dates=True)
        cols = df.columns
        # print(df.index.nlevels, cols, df.shape)

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
        if hasattr(df, "to_frame"):
            df = df.to_frame()
        if is_scalar(df):
            return Panel(df)
        else:
            df.columns = cols[: len(df.columns)]
            return Panel(pd.DataFrame(df).iloc[:, [0]].dropna())

    def trend(self, func: Callable, reference: Union["Panel", List] = None, fill_value: Union[int, float] = 0,
              how="left", interval: int = 0, **kwargs) -> "Panel":
        """Apply a function to compute a trend by stock over time
        Arguments:
            func: function to apply to each stock time series, must accept a Series and return a Series
            reference: optional Panel or List of Panels to join with before applying the function
            fill_value: value to fill missing values in the reference Panel
            how: type of join to perform with the reference Panel, default is 'left'
            interval: frequency to resample as number of months; 0 (default) is no resampling.
            kwargs: additional keyword arguments to pass to the function
        """
  
        def _resample(x, rule_, func_, **kwargs) -> pd.Series:
            """wrapper to resample a Series x by rule_ and apply func_"""
            if x.index.nlevels > 1:
                x = x.reset_index(level=1, drop=True)  # drop stock level
            orig_index = x.index
            x = x.resample(rule_).asfreq()  # ensure end-of-month frequency
            y = func_(x, **kwargs)
            y = y.reindex(orig_index)  # reindex to original dates
            return y
        
        # assert self.nlevels > 0, "Cannot apply function to scalar empty Panel"
        if not isinstance(reference, list):
            df = self.join_frame(reference, fill_value=fill_value, how=how, require_dates=True)
        else:
            df = self.frame.copy() if self.nlevels > 0 else pd.DataFrame()
            for ref in reference:
                df = self.join_frame(ref, fill_value=fill_value, how=how, require_dates=True)

        if interval:
            interval = pd.offsets.MonthEnd(interval)
        
        # Input has single index level
        if df.index.nlevels == 1:
            if interval:    # apply function with resampling
                df = _resample(df, rule_=interval, func_=func, **kwargs)
            else:
                df = func(df, **kwargs)  # apply function directly

        # Input has two index levels
        else:
            # apply function to each group by the first index level (date)
            if interval:
                df = df.groupby(level=1).apply(_resample, rule_=interval, func_=func, **kwargs)
            else:
                df = df.groupby(level=1).apply(func, **kwargs)

            # flatten extra index levels
            while df.index.nlevels > 2:
                df = df.reset_index(level=0, drop=True)

        # if the result is a Series, convert to DataFrame
        if hasattr(df, "to_frame"):
            df = df.to_frame()
        if is_scalar(df):
            return Panel(df)
        else:
            return Panel(pd.DataFrame(df).iloc[:, [0]].dropna())

    #
    # Panel ByGroup Operations
    #
    def __matmul__(self, other: "Panel") -> "Panel":
        """Compute the dot product of two Panels, by first index level date group."""

        def dot(x):
            """Dot product of two columns"""
            return (x.iloc[:, 0] * x.iloc[:, -1]).sum()

        return self.apply(dot, other, how="inner")

    #
    # Panel Advanced Operations
    #
    def shift(self, shift: int = 1) -> "Panel":
        """
        Shift (relabel) the date index of this Panel by a fixed number of periods.

        Use this when you want a **pure date shift** of existing rows—i.e., move each row’s
        date backward/forward by `shift` periods—without changing the panel’s frequency
        or the content of the rows.

        What this does
        --------------
        - Replaces each row’s date with a shifted date according to the Calendar’s mapping.
        - Preserves the original row granularity (no resampling).
        - Does **not** aggregate observations, forward-fill values, or create new rows.
        - This operation is a **date relabeling**, not a resample:
          it does not align data to month-ends or other lower-frequency targets.

        Parameters
        ----------
        shift : int, default 1
           Number of periods to shift the date index.
           Positive values shift dates forward; negative values shift dates backward

        Returns
        -------
        Panel
            A new Panel with the same data rows but with dates shifted by `shift` periods.
        """
        if self.nlevels > 0 and DATE_NAME in self._frame.index.names:
            out_panel = self.copy()
            nlevels = self.nlevels
            df = self.frame.reset_index(inplace=False)

            # Create dictionary to map original dates to shifted dates
            cal = Calendar(start_date=None, end_date=None)
            date_map = cal.dates_shifted(shift=shift)

            # drop rows with dates that cannot be shifted
            df = df[df[DATE_NAME].isin(date_map)]

            # Replace dates using the mapping dictionary
            df[DATE_NAME] = df[DATE_NAME].map(date_map)

            # Re-set the index and re-sort
            out_panel._frame = df.set_index(
                [DATE_NAME, STOCK_NAME][:nlevels]
            ).sort_index(level=list(range(nlevels)))
        else:
            out_panel = self.copy()
        return out_panel

    def restrict(
        self,
        min_value: float = None,
        max_value: float = None,
        start_date: str = None,
        end_date: str = None,
        mask: "Panel" = None,
        subset: "Panel" = None,
        min_stocks: int = None,
    ) -> "Panel":
        """Keeps the rows of this Panel based on date, stock, and value criteria.
        Arguments:
            start_date: Optional start date to filter the DataFrame (inclusive)
            end_date: Optional end date to filter the DataFrame (inclusive)
            min_stocks: Optional minimum number of stocks per date to keep the date
            min_value: Optional minimum value to keep the row
            max_value: Optional maximum value to keep the row
            mask: Optional Panel of boolean values to filter the DataFrame
            subset: Optional Panel whose index to keep in the DataFrame
        Returns:
            Panel with the filtered data
        """
        if self.nlevels < 1:  # empty or scalar
            return self.copy()
        df = self._frame.copy()
        ### start_date = str_or_None(start_date)
        if start_date and DATE_NAME in df.index.names:
            start_date = pd.to_datetime(start_date)
            df = df[df.index.get_level_values(DATE_NAME) >= start_date]
        ### end_date = str_or_None(end_date)
        if end_date and DATE_NAME in df.index.names:
            end_date = pd.to_datetime(end_date)
            df = df[df.index.get_level_values(DATE_NAME) <= end_date]
        ### min_value = numeric_or_None(min_value)
        if min_value is not None:
            df = df[df.iloc[:, 0] >= min_value]
        ### max_value = numeric_or_None(max_value)
        if max_value is not None:
            df = df[df.iloc[:, 0] <= max_value]
        if isinstance(mask, Panel):  # and mask.nlevels == self.nlevels:
            mask_df = mask.frame
            # if df.index.nlevels != mask_df.index.nlevels:
            #     raise ValueError("Cannot apply mask Panel with different index levels")
            df = df.join(mask_df, how="inner", rsuffix="_mask")
            df = df[df.iloc[:, -1].astype(bool)]  # keep only rows where mask is True
            df = df.iloc[:, :-1]  # drop the mask column
        if isinstance(subset, Panel):  #  and index.nlevels == self.nlevels:
            # only keep indexes that are in subset.frame
            index_df = subset.frame
            # print('index_df', index_df) ###
            # if df.index.nlevels != index_df.index.nlevels:
            #    raise ValueError("Cannot apply index Panel with different index levels")
            # print('df', df) ###
            df = df[df.index.isin(index_df.index)]
            # print('df', df) ###
            #df = df.join(index_df, how="inner", rsuffix="_index")
            #df = df.iloc[:, :-1]  # drop the index column
        ### min_stocks = numeric_or_None(min_stocks)
        if is_numeric_dtype(min_stocks) and self.nlevels == 2:
            counts = df.groupby(level=0).size()
            valid_dates = counts[counts >= min_stocks].index
            df = df[df.index.get_level_values(0).isin(valid_dates)]
        out_panel = Panel(df)
        out_panel.name = self.name
        return out_panel

    def plot(self, other_panel: "Panel" = None, **kwargs):
        """Plot the values of this Panel.
        Arguments:
            other_panel: Optional other Panel to plot on the same axes
            kwargs: keyword arguments to pass to pandas.DataFrame.plot()
        """
        df = self.frame.copy()
        if other_panel is not None:
            df = df.join(other_panel.frame, how="outer", rsuffix="_2")
            if "x" not in kwargs:
                kwargs["x"] = df.columns[0]
            if "y" not in kwargs:
                kwargs["y"] = df.columns[1]
        df.plot(**kwargs)


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

def write_report(signal: Panel) -> str:
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


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("error", category=RuntimeWarning)
    tic = time.time()
    print(str(datetime.now()))

    #
    # Helpers to display Panels
    #

    def panel_info(panel: Panel) -> Dict[str, Any]:
        """Return basic information about this Panel."""
        info = {"nlevels": panel.nlevels, "rows": len(panel)}
        info["memory_usage_bytes"] = (
            0 if panel.nlevels < 0 else int(panel.frame.memory_usage(deep=True).sum())
        )
        if panel.nlevels >= 1:
            dates = panel.dates
            info["num_dates"] = len(dates)
            info["min_date"] = str(dates[0])[:10]
            info["max_date"] = str(dates[-1])[:10]
        if panel.nlevels == 2:
            info["max_stocks_per_date"] = int(panel.frame.groupby(level=0).size().max())
            info["min_stocks_per_date"] = int(panel.frame.groupby(level=0).size().min())
        return info

    def frame_info(frame: pd.DataFrame) -> Dict[str, Any]:
        """Return counts by date if a DataFrame."""
        return frame.groupby(level=0).size().to_dict()

    def show(x: str | int | Panel):
        """Shows summary of dataframe from Panel named x or _x"""
        if isinstance(x, int):
            x = Panel().load(f"_{x}")
        elif isinstance(x, str):
            if not x.startswith("_") and x.isdigit():
                x = "_" + x
            x = Panel().load(x)
        print(x.frame)
        print(str(x))

    def p(x: int | str):
        """Load Panel named x or _x"""
        if isinstance(x, int):
            return Panel().load(f"_{x}")
        elif isinstance(x, str):
            if not x.startswith("_") and x.isdigit():
                x = "_" + x
            return Panel().load(x)
        else:
            raise ValueError("Input must be int or str")

    dates = DATES
    print(json.dumps(dates, indent=4))

    #from server_utils import str_or_None, panel_or_numeric
 
    toc = time.time()
    print(f"Total elapsed time: {toc - tic:.2f} seconds")
    print(str(datetime.now()))
