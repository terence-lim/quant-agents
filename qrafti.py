# qrafti.py  (c) Terence Lim

from utils import DataCache, Calendar, CRSP_RAG_PATH, JKP_RAG_PATH
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict, Union, Set, Any, Tuple, Callable
from datetime import datetime
import time
import pandas as pd
from pandas.api.types import is_list_like, is_scalar, is_numeric_dtype, is_float_dtype
import logging
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
logging.disable(logging.DEBUG)
#pd.set_option("future.no_silent_downcasting", True)  # for fillna behavior

STOCK_NAME = "permno"
DATE_NAME = "eom"

#
# These are temporarily here, perhaps should be in utils.py...
#
#DATES = dict(start_date="2020-01-01", end_date="2024-12-31")
#DATES = dict(start_date="2001-01-01", end_date="2024-12-31")
DATES = dict(start_date="1993-01-01", end_date="2024-12-31")

CRSP_VERSION = True
if CRSP_VERSION:
    RAG_PATH = CRSP_RAG_PATH
else:
    RAG_PATH = JKP_RAG_PATH


###########################
#
# Panel Data Structure and Operations
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



if __name__ == "__main__":
    import warnings
    from utils import OUTPUT
    warnings.filterwarnings("error", category=RuntimeWarning)
    tic = time.time()
    print(str(datetime.now()))

    #
    # Helpers to display Panels
    #

    def evaluate_panels(panel: Panel, ground: Panel, experiment: str = '') -> pd.DataFrame:
        """Evaluate two Panels side by side"""
        if panel.nlevels != ground.nlevels or panel.nlevels < 1:
            return None
        output = dict()
        output['panel_dates'] = len(panel.dates)
        output['ground_dates'] = len(ground.dates)    
        output['panel_rows'] = len(panel)
        output['ground_rows'] = len(ground)
        both = panel.frame.dropna().join(ground.frame.dropna(), how="inner", rsuffix="_ground")
        output['both_rows'] = len(both)
        output['both_dates'] = len(both.index.get_level_values(0).unique())
        output['spearman'] = both.corr(method="spearman").iloc[0, 1]
        output['pearson'] = both.corr(method="pearson").iloc[0, 1]
        # difference of means, divided by average standard deviation
        output['diff_stdz'] = ((both.iloc[:, 0].mean() - both.iloc[:, 1].mean()) / 
                              ((both.iloc[:, 0].std() + both.iloc[:, 1].std()) / 2))
        with open(OUTPUT / 'evaluation.csv', 'a') as f:
            if f.tell() == 0:
                f.write("experiment & panel_dates & ground_dates & both_dates & panel_rows & ground_rows & both_rows & spearman & pearson & diff_stdz\n")
            f.write(f"{experiment} & {output['panel_dates']} & {output['ground_dates']} & {output['both_dates']} & "
                    f"{output['panel_rows']} & {output['ground_rows']} & {output['both_rows']} & "
                    f"{output['spearman']:.4f} & {output['pearson']:.4f} & {output['diff_stdz']:.4f}\n")
        return pd.DataFrame(output, index=[experiment])
    
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

    toc = time.time()
    print(f"Total elapsed time: {toc - tic:.2f} seconds")
    print(str(datetime.now()))
