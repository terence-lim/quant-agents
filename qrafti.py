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
from pandas.api.types import is_list_like, is_integer_dtype, is_scalar, is_numeric_dtype
import matplotlib.pyplot as plt
import warnings

from portfolio import PortfolioEvaluation
pd.set_option('future.no_silent_downcasting', True) # for fillna behavior

DATA_LAKE = Path('/home/terence/Downloads/scratch/2024/JKP/')
WORKSPACE = DATA_LAKE / 'workspace'
MEDIA = DATA_LAKE / 'media'

STOCK_NAME = 'permno'
DATE_NAME = 'eom'
UNIVERSE_PANEL = 'SIZE_DECILE' # 'ret_exc_lead1m'
###########################
#
# Data Cache library for persisting intermediate DataFrames
#
###########################
class DataCache:

    @staticmethod
    def load_cache() -> Dict[str, str]:
        """Load the data cache from the cache file."""
        cache_file = Path(WORKSPACE / 'cache.json')
        try:
            with open(cache_file, 'rb') as f:
                cache = json.load(f)
        except:
            cache = {"file_id": 0}
        return cache
    
    @staticmethod
    def dump_cache(cache: Dict[str, str]):
        """Dump the data cache to the cache file."""
        cache_file = Path(WORKSPACE / 'cache.json')
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=4)

    @staticmethod
    def write_frame(frame: pd.DataFrame, name: str = '', folder: str = WORKSPACE) -> str:
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
        if frame is not None:
            frame.to_parquet(folder / f"{name}.parquet", index=True)
        return name

    @staticmethod
    def read_frame(name: str, folder: str = WORKSPACE) -> pd.DataFrame:
        """Read a dataframe from parquet file in the data cache
        Arguments:
            name: Name of the file (without extension)
            folder: Folder to read the parquet files from
        Returns:
            frame: DataFrame read from the file, None if not found
        """
        filename = folder / f"{name}.parquet"
        if filename.exists():
            return pd.read_parquet(filename)
        else:
            return None

    @staticmethod
    def reset(cache_path = WORKSPACE):
        """Clears and resets data cache"""
        for file in cache_path.glob("_*.parquet"):
            file.unlink()
        cache_file = Path(WORKSPACE / 'cache.json')
        if cache_file.exists():
            cache_file.unlink()

#
# TO DO: as RAG
#
def load_variables(filename = 'variables.txt', 
                   data_path: Path = DATA_LAKE) -> pd.DataFrame:
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

#
# Running code in the Data Cache 
#
import subprocess, sys, os#
def run_code_in_subprocess(code_str):
    env = os.environ.copy()
    # prepend your project root to PYTHONPATH
    env["PYTHONPATH"] = "/home/terence/Dropbox/github/thesis/quant-agents:" + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, "-c", code_str],
        capture_output=True,
        text=True,
        env=env
    )
    print(f"Subprocess exited with code {proc.returncode}")
    return proc.stdout, proc.stderr, proc.returncode    


###########################
#
# Calendar of valid dates
#
###########################
class Calendar:
    def __init__(self, start_date: str = '', end_date: str = '', reference_panel: str = UNIVERSE_PANEL):
        # Initialize the Calendar with unique sorted dates from a reference PanelFrame 'ret_exc_lead1m'
        dates = PanelFrame(reference_panel).frame.index.get_level_values(0)
        if start_date:
            dates = dates[dates >= start_date]
        if end_date:
            dates = dates[dates <= end_date]

        sorted_dates = sorted(dates.unique().tolist())
        self.start_date = sorted_dates[0]
        self.end_date = sorted_dates[-1]
        self.dates = pd.Series(range(len(sorted_dates)), index=sorted_dates)

    def as_dates(self, dates: List) -> List[str]:
        """Converts a list of dates to valid dates in the calendar."""
        dates_dict = {d[:4]+d[5:7]: d for d in self.dates.index}
        dates_dict |= {int(d): date for d, date in dates_dict.items()}
        dates_dict |= {int(d[:4]+d[5:7]+d[8:10]): d for d in self.dates.index}
        #dates_dict |= {d:d for d in self.dates.index}
        return [dates_dict.get(d, str(d)) for d in dates]

    def dates_shifted(self, shift: int = 1) -> Dict[str, str]:
        """Return a mapping of original dates to shifted dates."""
        shifted_dates = dict()
        for i in range(len(self.dates)):
            j = i + shift
            if 0 <= j < len(self.dates):
                shifted_dates[self.dates.index[i]] = self.dates.index[j]
        return shifted_dates
    
    def dates_range(self, start_date: str, end_date: str) -> List[str]:
        """Return a list of dates in the calendar between start_date and end_date (inclusive)."""
        return [date for date in self.dates.index if start_date <= date <= end_date]
    
    def ismonth(self, date: str, months : List | int) -> bool:
        """Check if a date is in the specified months."""
        if isinstance(months, int):
            months = [months]
        month = int(date[5:7])
        return month in months
    
    def offset(self, date: str, offset: int, strict: bool) -> str:
        """Return the date offset by the specified number of periods."""
        if date in self.dates.index:
            i = self.dates[date] + offset
            if 0 <= i < len(self.dates):
                return self.dates.index[i]
        if not strict:
            return self.dates.index[0 if i < 0 else -1]
        return ''

def frame_info(frame: pd.DataFrame) -> Dict[str, Any]:
    """Return basic information about a DataFrame."""
    return frame.groupby(level=0).size().to_dict()


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

    def __init__(self, name: str = '', start_date: str = '', end_date: str = ''):
        """Initialize a PanelFrame, optionally from a cached DataFrame file and date range.

        Arguments:
            name: Optionally load from named cached DataFrame file (without extension)
            start_date: Optional start date to filter the DataFrame (inclusive)
            end_date: Optional end date to filter the DataFrame (inclusive)
        """
        self.name = name
        if name:
            self._frame = DataCache.read_frame(name)  # may be None
            if self.nlevels > 0:
                if start_date:
                    self._frame = self._frame[self._frame.index.get_level_values(0) >= start_date]
                if end_date:
                    self._frame = self._frame[self._frame.index.get_level_values(0) <= end_date]
        else:
            self._frame = None

    def __len__(self) -> int:
        """Return the number of data items in this PanelFrame."""
        return 0 if self._frame is None else len(self._frame)

    @property
    def frame(self) -> pd.DataFrame:
        """Return the underlying DataFrame or scalar value of this PanelFrame."""
        if self.nlevels == 0:
            return self._frame.iloc[0,0]
        return self._frame

    @property
    def info(self) -> Dict[str, Any]:
        """Return basic information about this PanelFrame."""
        info = {'nlevels': self.nlevels, 'rows': len(self)}
        if self.nlevels >= 1:
            dates = self.dates
            info['num_dates'] = len(dates)
            info['min_date'] = str(dates[0])
            info['max_date'] = str(dates[-1])
        if self.nlevels == 2:
            info['max_stocks_per_date'] = int(self.frame.groupby(level=0).size().max())
            info['min_stocks_per_date'] = int(self.frame.groupby(level=0).size().min())
        info['memory_usage_bytes'] = int(self.frame.memory_usage(deep=True).sum())
        return info

    @property
    def nlevels(self) -> int:
        """Number of index levels of this PanelFrame
        Returns:
            -1 if empty, 0 if scalar, 1 if single index level (date), 2 if multi-index (date, stock)
        """
        if self._frame is None:
            return -1
        elif self._frame.index.nlevels == 1 and self._frame.index.name not in [DATE_NAME, STOCK_NAME]:
            return 0
        else:
            return self._frame.index.nlevels

    @property
    def values(self) -> np.ndarray:
        """Return the values of this PanelFrame as a numpy array.
        Returns:
            [] if empty, 1D array otherwise
        """
        return [] if self.nlevels < 0 else self.frame.iloc[:, 0].values
        
    @property
    def dates(self) -> List[str]:
        """Return the list of unique dates in this PanelFrame."""
        return [] if self.nlevels <= 0 else sorted(self.frame.index.get_level_values(0).unique())

    def __getitem__(self, key: Union[str, int, Tuple]) -> Any:
        """Get item(s) from this PanelFrame using indexing.
        Arguments:
            key: Can be a date string, integer index, or tuple of (date, stock)
        Returns:
            The value(s) at the specified index, or None if not found
        """
        try:
            if self.nlevels == 1:
                if isinstance(key, str):
                    return self.frame.loc[key].iloc[0, 0]
                else: # isinstance(key, int):
                    return self.frame.iloc[key, 0]
            else: # nlevels == 2
                if isinstance(key, tuple) and len(key) == 2:  # (date, stock)
                    return self.frame.loc[key].iloc[0, 0]
                elif isinstance(key, str):  # return all stocks for that date
                    return self.frame.xs(key, level=0).iloc[:, 0]
                else:  # return all dates for that stock
                    return self.frame.xs(key, level=1).iloc[:, 0]
        except Exception as e:
            return None

    #
    # Primitive helpers
    #
    def join_frame(self, other: 'PanelFrame', fill_value: Any, how: str) -> pd.DataFrame:
        """Helper to join columns from another PanelFrame, and return as a DataFrame
        Arguments:
            other: Another PanelFrame to join with, or a scalar value to add as a column
            fill_value: Value to fill missing values in the other PanelFrame
            how: Type of join to perform ('left', 'right', 'inner', 'outer')
        Returns:
            df: DataFrame with the joined data
        """
        assert (self.nlevels > 0) or (isinstance(other, PanelFrame) and other.nlevels > 0), "At least one PanelFrame with index levels"
        if self.nlevels > 0:
            # self is a multi-index PanelFrame:
            df = self.frame.copy()
        else:
            # self is a scalar PanelFrame: create DataFrame with same index as other
            data = fill_value if self._frame is None else self.frame
            df = pd.DataFrame(index=other.frame.index, data=data, columns=['self'])

        if isinstance(other, PanelFrame) and other.nlevels >= 0:  # other is a PanelFrame
            if other.nlevels > 0:
                # other is also a multi-index PanelFrame: join on index levels
                other_df = other.frame
                assert df.index.nlevels == other_df.index.nlevels, "Cannot join PanelFrames with different index levels"
                df = df.join(other_df, how=how, rsuffix='_').fillna(fill_value)
            else:
                # other is a scalar PanelFrame: add as a column with same value
                df['other'] = other.frame
        else:  # other is a scalar or None
            if other is not None:    # MAY REMOVE THIS LINE
                df['other'] =  other
        return df


    #
    # Primitive operations
    #
    def copy(self) -> 'PanelFrame':
        """Return a copy of this PanelFrame.
        Arguments:
            deep: If True, also copy the underlying DataFrame, otherwise just copy the metadata
        Returns:
            A new PanelFrame with the same name, date range, and optionally a copy of the DataFrame
        """
        new_panel = PanelFrame()
        new_panel.name = self.name
        new_panel._frame = None if self.nlevels < 0 else self._frame.copy()
        return new_panel
        
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

        def _frame_is_scalar(frame: pd.DataFrame) -> bool:
            """Helper to check if a DataFrame should be considered a scalar (1x1)"""
            if frame.index.nlevels == 1 and frame.index.name not in [DATE_NAME, STOCK_NAME]:
                assert len(frame) == 1, "Only single row allowed for scalar"
                return True
            return False

        if isinstance(frame, pd.Series):
            frame = frame.to_frame()
        if is_scalar(frame):
            self._frame = _scalar_as_frame(frame)
        else:
            if isinstance(frame, pd.DataFrame):
                if _frame_is_scalar(frame):
                    self._frame = _scalar_as_frame(frame)
                elif append and self.frame is not None:
                    old_frame = self.frame
                    frame.columns = old_frame.columns # force new column names to match existing

                    # drop duplicates based on index, keep the last occurrence
                    self._frame = pd.concat([old_frame, frame], axis=0).sort_index(level=range(frame.index.nlevels))
                    self._frame = self.frame[~self.frame.index.duplicated(keep='last')]
                else:
                    self._frame = frame.sort_index(level=range(frame.index.nlevels))
                    self._frame.index.names = [DATE_NAME, STOCK_NAME][:frame.index.nlevels] # ensure index names
            else:
                assert False, "Frame must be a pandas DataFrame or scalar"
        return self

    def astype(self, dtype) -> 'PanelFrame':
        """Change the dtype of the values of this PanelFrame."""
        if self.nlevels >= 0:
            self._frame = self._frame.astype(dtype)
        return self

    def shift(self, shift: int = 1) -> 'PanelFrame':
        """Shift the dates of this PanelFrame"""
        if self.nlevels <= 0:  # empty or scalar
            out_panel = self.copy()
        else:  # nlevels == 1 or 2
            out_panel = self.copy()
            nlevels = self.nlevels
            df = self.frame.reset_index(inplace=False)

            # Create dictionary to map original dates to shifted dates
            cal = Calendar()
            date_map = cal.dates_shifted(shift=shift)

            # drop rows with dates that cannot be shifted
            df = df[df[DATE_NAME].isin(date_map)]

            # Replace dates using the mapping dictionary
            df[DATE_NAME] = df[DATE_NAME].map(date_map)

            # Re-set the index and re-sort
            out_panel._frame = df.set_index([DATE_NAME, STOCK_NAME][:nlevels]).sort_index(level=list(range(nlevels)))
        return out_panel

    def persist(self, name: str = '') -> 'PanelFrame':
        """Set this PanelFrame to persist its data to cache file.
        Arguments:
            name: Optional name for the file, if not given, a new name will be generated
        Returns:
            self: This PanelFrame
        """
        name = DataCache.write_frame(frame=self._frame, name=name)
        self.name = name
        return self
        
    #
    # PanelFrame Binary Operators
    #
    def _operands(self, other: 'PanelFrame', fill_value: Any, how: str) -> Tuple[pd.Series, pd.Series]:
        """Internal helper to align another PanelFrame or scalar to this PanelFrame
        Notes: If other is a PanelFrame, perform an outer join, and fill missing values with fill_value
        """
        if isinstance(other, PanelFrame):
            df = self.join_frame(other, fill_value=fill_value, how=how)
            df_other = df.iloc[:, 1]
            df = df.iloc[:, 0]
        else:
            if self.nlevels > 0:
                df = self._frame.iloc[:, 0]
            else:
                df = fill_value if self._frame is None else self.frame
            df_other = other
        return df, df_other

    def __add__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Add values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=0, how='outer')
        return PanelFrame().set_frame(df + df_other)

    def __radd__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Add values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=0, how='outer')
        return PanelFrame().set_frame(df_other + df)
    
    def __sub__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Subtract values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=0, how='outer')
        return PanelFrame().set_frame(df - df_other)

    def __rsub__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Subtract values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=0, how='outer')
        return PanelFrame().set_frame(df_other - df)
    
    def __mul__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Multiply values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=1, how='outer')
        return PanelFrame().set_frame(df * df_other)
    
    def __rmul__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Multiply values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=1, how='outer')
        return PanelFrame().set_frame(df_other * df)
    
    def __truediv__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Divide values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=1, how='outer')
        return PanelFrame().set_frame(df / df_other)
    
    def __rtruediv__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Divide values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=1, how='outer')
        return PanelFrame().set_frame(df_other / df)

    def __eq__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Check equality of values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how='inner')
        return PanelFrame().set_frame((df == df_other).astype(bool))
    
    def __ge__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Check equality of values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how='inner')
        return PanelFrame().set_frame((df >= df_other).astype(bool))

    def __gt__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Check equality of values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how='inner')
        return PanelFrame().set_frame((df > df_other).astype(bool))

    def __le__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Check equality of values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how='inner')
        return PanelFrame().set_frame((df <= df_other).astype(bool))

    def __lt__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Check equality of values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how='inner')
        return PanelFrame().set_frame((df < df_other).astype(bool))

    def __ne__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Check inequality of values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how='inner')
        return PanelFrame().set_frame((df != df_other).astype(bool))

    def __or__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Logical or of values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=0, how='outer')
        return PanelFrame().set_frame((df.astype(bool) | df_other.astype(bool)))

    def __and__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Logical or of values of this PanelFrame with other"""
        df, df_other = self._operands(other, fill_value=1, how='inner')
        return PanelFrame().set_frame((df.astype(bool) & df_other.astype(bool)))

    #
    # PanelFrame Unary Operators
    #
    def __neg__(self) -> 'PanelFrame':
        """Negate the values of this PanelFrame."""
        return PanelFrame().set_frame(-self.frame)
    
    def __invert__(self) -> 'PanelFrame':
        """Boolean negation of the values of this PanelFrame."""
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

    def filter(self, min_value: float = None, max_value: float = None, values: List = None,
               start_date: str = None, end_date: str = None, dates: List[str] = None, 
               dropna: bool = False, mask: 'PanelFrame' = None, index: 'PanelFrame' = None,
               stocks: List[int] = None, min_stocks: int = None) -> 'PanelFrame':
        """Filter the values of this PanelFrame based on date, stock, and value criteria.
        Arguments:
            start_date: Optional start date to filter the DataFrame (inclusive)
            end_date: Optional end date to filter the DataFrame (inclusive)
            dates: Optional list of dates to filter the DataFrame
            stocks: Optional list of stocks to filter the DataFrame
            min_stocks: Optional minimum number of stocks per date to keep the date
            min_value: Optional minimum value to keep the row
            max_value: Optional maximum value to keep the row
            values: Optional list of values to keep the row
            mask: Optional PanelFrame of boolean values to filter the DataFrame
            index: Optional PanelFrame to filter index to
            dropna: If True, drop rows with NaN values
        Returns:
            PanelFrame with the filtered data
        """
        out_panel = self.copy()
        if self.nlevels < 1:   # empty or scalar
            return out_panel
        df = self.frame
        if start_date:
            df = df[df.index.get_level_values(0) >= start_date]
        if end_date:
            df = df[df.index.get_level_values(0) <= end_date]
        if dates:
            df = df[df.index.get_level_values(0).isin(dates)]
        if stocks:
            df = df[df.index.get_level_values(1).isin(stocks)]
        if min_value is not None:
            df = df[df.iloc[:, 0] >= min_value]
        if max_value is not None:
            df = df[df.iloc[:, 0] <= max_value]
        if values is not None:
            df = df[df.iloc[:, 0].isin(values)]
        if dropna:
            df = df[df.iloc[:, 0].notna()]
        if isinstance(mask, PanelFrame) and mask.nlevels == self.nlevels:
            mask_df = mask.frame
            if df.index.nlevels != mask_df.index.nlevels:
                raise ValueError("Cannot apply mask PanelFrame with different index levels")
            df = df.join(mask_df, how='inner', rsuffix='_mask')
            df = df[df.iloc[:, -1].astype(bool)]  # keep only rows where mask is True
            df = df.iloc[:, :-1]  # drop the mask column
        if isinstance(index, PanelFrame) and index.nlevels == self.nlevels:
            # only keep indexes that are in index.frame
            index_df = index.frame
            if df.index.nlevels != index_df.index.nlevels:
                raise ValueError("Cannot apply index PanelFrame with different index levels")
            df = df.join(index_df, how='inner', rsuffix='_index')
            df = df.iloc[:, :-1]  # drop the index column
        if is_numeric_dtype(min_stocks) and self.nlevels == 2:
            counts = df.groupby(level=0).size()
            valid_dates = counts[counts >= min_stocks].index
            df = df[df.index.get_level_values(0).isin(valid_dates)]
        out_panel._frame = df
        return out_panel

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
    def trend(self, func: Callable) -> 'PanelFrame':
        """Apply a function to compute a trend by stock over time"""
        assert self.nlevels == 2, "Trend can only be computed for PanelFrames with 2 index levels"
        df = self.frame.reset_index().sort_values(by=[STOCK_NAME, DATE_NAME])
        df['trend'] = df.groupby([STOCK_NAME]).apply(func, include_groups=False).values
        df = df.set_index([DATE_NAME, STOCK_NAME]).sort_index()
        return PanelFrame().set_frame(pd.DataFrame(df).iloc[:, [-1]])
    
    def fill(self, *panels, replace: List = []) -> 'PanelFrame':
        """Replace specified values in the first column with values from other PanelFrames in order

        Arguments:
            panels: Other PanelFrames to use for filling values
            replace: List of values to be replaced in the first column
        Returns:
            PanelFrame with the filled values
        """
        def replace_helper(x, replace: List) -> pd.Series:
            """Helper to replace NaN or listed values in the first column with values from the second column"""
            x.iloc[:, 0] = x.iloc[:, 0].fillna(x.iloc[:, 1])
            mask = x.iloc[:, 0].isin(replace)
            x.loc[mask, x.columns[0]] = x.loc[mask, x.columns[1]]
            return x.iloc[:, 0]
    
        if not is_list_like(replace):
            replace = [replace]
        out_panel = self.copy()
        for panel in panels:
            out_panel = out_panel.apply(replace_helper, panel, how='outer', fill_value=np.nan, replace=replace)
        return out_panel


    def apply(self, func: Callable, reference: 'PanelFrame' = None, fill_value=0, how='left', **kwargs) -> 'PanelFrame':
        """Apply a function to each date group of PanelFrame, optionally based on values reference PanelFrame.
        Arguments:
            func: function to apply to each date group, must accept a DataFrame and return a Series
            reference: optional PanelFrame to join with before applying the function
            fill_value: value to fill missing values in the reference PanelFrame
            how: type of join to perform with the reference PanelFrame, default is 'left'
            kwargs: additional keyword arguments to pass to the function
        Returns:
            PanelFrame with the same index as this PanelFrame, with values computed by the function
        """
        assert self.nlevels > 0, "Cannot apply function to scalar empty PanelFrame"
        df = self.join_frame(reference, fill_value=fill_value, how=how)
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
        if is_scalar(df):
            return PanelFrame().set_frame(df)
        else:
            df.columns = cols[:len(df.columns)]
            return PanelFrame().set_frame(pd.DataFrame(df).iloc[:, [0]])

    def __matmul__(self, other: 'PanelFrame') -> 'PanelFrame':
        """Compute the dot product of two PanelFrames, by first index level date group."""
        def dot(x):
            """Dot product of two columns"""
            return (x.iloc[:, 0] * x.iloc[:, -1]).sum()
        return self.apply(dot, other, how='inner')

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
        panel_frame.apply(weighted_average, weights or 1, fill_value=0)
    """
    return (x.iloc[:, 0] * x.iloc[:, 1]).sum() / x.iloc[:, 1].sum()


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
    print(x)
    lower, upper = x.loc[x.iloc[:,1].astype(bool), x.columns[0]].quantile([lower, upper]).values
    return x.iloc[:, 0].clip(lower=lower, upper=upper)


def digitize(x, cuts: int | List[float], ascending: bool = True) -> pd.Series:
    """
    Discretize the first column based on rows that are True in the last column.
    Arguments:
        x: DataFrame with at least two columns, first column is the data to be discretized,
           last column is a boolean indicator for which rows to consider for discretizing
        cuts: If int, number of quantile-based bins to create; if list, the breakpoints to use
        ascending: If True (default), lower values get lower bin numbers
    Returns:
        pd.Series with discretized bin assignments (1, 2, ..., cuts) for each row in the first column
    Usage:
        panel_frame.apply(digitize, indicator or True, fill_value=False, cuts=cuts

    """
    if is_list_like(cuts):
        q = np.concatenate([[0], cuts, [1]])
    else:
        q = np.linspace(0, 1, cuts + 1)
    breakpoints = x.loc[x.iloc[:,1].astype(bool), x.columns[0]].quantile(q=q).values
    breakpoints[0] = -np.inf            
    breakpoints[-1] = np.inf
    ranks = pd.cut(x.iloc[:,0], bins=breakpoints, labels=range(1, len(breakpoints)), include_lowest =True)
    if not ascending:
        ranks = len(breakpoints) - ranks.astype(int) + 1
    return ranks.astype(int)

def portfolio_weights(x, leverage: float = 1.0) -> pd.Series:
    """Scale the the weights to sum to the given leverage
    Arguments:
        x: DataFrame with weights
        leverage: Total leverage to scale the weights to
    Returns:
        pd.Series with the scaled weights
    Usage:
        panel_frame.apply(portfolio_weights, leverage=leverage)
    """
    total_weight = x.iloc[:, 0].abs().sum()
    if total_weight == 0:
        return x.iloc[:, 0].rename(x.columns[0])
    return x.iloc[:, 0].mul(abs(leverage)).div(total_weight).rename(x.columns[0])

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
        panel_frame.apply(spread_portfolios, weights or 1, fill_value=0)
    """
    low_quantile = x.iloc[:, 0].min()
    high_quantile = x.iloc[:, 0].max()
    other_quantile = ~x.iloc[:, 0].isin([low_quantile, high_quantile])
    x.iloc[:, 0] = x.iloc[:, 0].replace({low_quantile: -1, high_quantile: 1})
    x.iloc[other_quantile, 0] = 0
    x['_total_weight'] = x.groupby(x.columns[0])[x.columns[1]].transform('sum') # normalize weights
    return x.iloc[:, 0].mul(x.iloc[:, 1]).div(x['_total_weight']).rename(x.columns[0])

#
# Common functions to be used with PanelFrame.trend()
#
def cumcount(x) -> pd.Series:
    """
    Compute the cumulative count of observations
    Arguments:
        x: DataFrame with at least one column
    Returns:
        pd.Series with the cumulative count of observations
    Usage:
        panel_frame.trend(cumcount)
    """
    return pd.Series(np.arange(len(x)), index=x.index)

#
# Factor Functions
#
def factor_snapshots(factor: 'PanelFrame', month: List | int = []) -> 'PanelFrame':
    """Extract snapshot of the factor values at a specific months
    Arguments:
        factor: PanelFrame of factor values
        month: List of months (1-12) to extract snapshots for, if empty, extract for all months
    Returns:
        PanelFrame of factor values at the specified date, forward filled from previous dates
    """
    assert factor.nlevels == 2, "Factor must have two index levels"

    factor_dates = factor.dates
    prev_date = factor_dates[0]
    cal = Calendar(start_date=factor_dates[0], end_date=factor_dates[-1])
    snapshot_df = []
    for next_date in cal.dates_range(cal.start_date, cal.end_date):
        if not month or cal.ismonth(next_date, month):
            for curr_date in cal.dates_range(prev_date, next_date):
                if curr_date in factor_dates:
                    factor_df = factor.frame.xs(curr_date, level=0).reset_index()
                    factor_df[DATE_NAME] = next_date
                    factor_df['_date_'] = curr_date
                    snapshot_df.append(factor_df)
            prev_date = cal.offset(next_date, 1, strict=True)

    # sort by STOCK_NAME, DATE_NAME and _date_ and drop duplicates, keep last
    snapshot_final = pd.concat(snapshot_df, axis=0)
    snapshot_final = snapshot_final.sort_values(by=[STOCK_NAME, DATE_NAME, '_date_'])
    snapshot_final = snapshot_final.drop_duplicates(subset=[STOCK_NAME, DATE_NAME], keep='last')
    snapshot_final = snapshot_final.set_index([DATE_NAME, STOCK_NAME]).drop(columns=['_date_'])
    snapshot_panel = PanelFrame().set_frame(snapshot_final)
    return snapshot_panel

def factor_generate(factor: 'PanelFrame', lags: int, window: int, univ: 'PanelFrame' = None) -> 'PanelFrame':
    """Generate a factor PanelFrame from rolling windows based on universe filter.
    Arguments:
        lags: Number of months to lag the factor values
        window: Window size for rolling accumulation of factor values
        univ: Optional PanelFrame of universe filter
    Returns:
        PanelFrame of generated factor values
    """
    assert factor.nlevels == 2, "Factor must have two index levels"
    cal = Calendar()
    factor_dates = factor.dates
    start_date = cal.offset(factor_dates[0], offset=lags, strict=False)
    end_date = cal.offset(factor_dates[-1], offset=lags, strict=False)
    factor_final = []
    for next_date in cal.dates_range(start_date, end_date):
        # For each date, collect data from lagged window
        start_window = cal.offset(next_date, offset = -(window + lags), strict=False)
        end_window = cal.offset(next_date, offset = -lags, strict=False)
        for curr_date in cal.dates_range(start_window, end_window):
            if curr_date in factor_dates:
                factor_df = factor.frame.xs(curr_date, level=0).reset_index()
                factor_df[DATE_NAME] = next_date
                factor_df['_date_'] = curr_date
                factor_final.append(factor_df)

    # sort by STOCK_NAME, DATE_NAME and _date_ and drop duplicates, keep last
    factor_final = pd.concat(factor_final, axis=0)
    factor_final = factor_final.sort_values(by=[STOCK_NAME, DATE_NAME, '_date_'])
    factor_final = factor_final.drop_duplicates(subset=[STOCK_NAME, DATE_NAME], keep='last')
    factor_final = factor_final.set_index([DATE_NAME, STOCK_NAME]).drop(columns=['_date_'])

    # require index to be in univ.frame
    factor_final = factor_final.join(univ.frame, how='inner', rsuffix='_univ').iloc[:, :1]
    factor_final = PanelFrame().set_frame(factor_final)
    return factor_final


#
# Portfolio Functions
#
def portfolio_evaluation(port_returns: 'PanelFrame') -> Dict[str, float]:
    """Compute summary performance statistics of a portfolio given its weights and returns.
    Arguments:
        port_returns: PanelFrame of factor returns
    Returns:
        Dict of summary statistics: mean return, volatility, Sharpe ratio
    """
    return {} if port_returns.nlevels != 1 else PortfolioEvaluation(port_returns.frame).summary()

def portfolio_returns(port_weights: 'PanelFrame', 
                      stock_returns: 'PanelFrame' = None) -> 'PanelFrame':
    """Compute the portfolio returns given portfolio weights and stock returns.
    Arguments:
        port_weights: PanelFrame of portfolio weights
        stock_returns: PanelFrame of leading stock returns
    Returns:
        PanelFrame of portfolio returns, shifted by one date to align with end of holding period
    """
    if stock_returns is None:
    #    stock_returns = PanelFrame('ret_exc_lead1m')
        stock_returns = PanelFrame('RET').shift(-1)  # RET should be leading dates, to compute realized returns
    #retx = PanelFrame('ret_exc_lead1m').shift(1)
    retx = PanelFrame('RETX')   # RETX should be actual dates, for drifting previous weights only
    port_weights = portfolio_impute(port_weights, retx, normalize=True)
    return (port_weights @ stock_returns).shift(1)


def portfolio_impute(port_weights: 'PanelFrame', retx: 'PanelFrame',
                     normalize: bool = True, drifted: bool = False) -> 'PanelFrame':
    """Impute missing portfolio weights on missing dates by forward drifting previous weights.
    Arguments:
        port_weights: PanelFrame of portfolio weights
        retx: PanelFrame of stock returns to forward drift previous weights
        normalize: If True, re-normalize weights to be dollar-neutral after forward drifting
        drifted: Whether to output all drifted weights (True), or only fill in missing dates
    Returns:
        PanelFrame of portfolio weights with missing dates imputed by forward drifting
    Notes:
        Side effect: Changes port_weights in place where missing dates are added.
    """
    assert port_weights.nlevels == 2, "Portfolio weights must have two index levels"
    portfolio_dates = port_weights.dates
    cal = Calendar(start_date=portfolio_dates[0], end_date=portfolio_dates[-1])
    all_dates = cal.dates_range(cal.start_date, cal.end_date)
    if len(all_dates) == len(portfolio_dates) and not drifted:
        return port_weights  # no missing dates to impute

    # pre-compute long and short notional based on first date
    long_notional = port_weights.frame.xs(portfolio_dates[0], level=0)
    long_notional = long_notional[long_notional > 0].sum().abs().iloc[0]
    short_notional = port_weights.frame.xs(portfolio_dates[0], level=0)
    short_notional = short_notional[short_notional < 0].sum().abs().iloc[0]

    prev_weights = None
    drifted_weights = []
    for date in tqdm(all_dates):
        if (drifted or date not in portfolio_dates) and prev_weights is not None:
            # forward drift previous weights if any
            if (retx is not None and date in retx.frame.index.get_level_values(0)):
                # using retx returns to drift previous weights
                returns = retx.frame.xs(date, level=0).reindex(prev_weights.index, fill_value=0)
                curr_weights = (prev_weights.iloc[:,0] * (1 + returns.iloc[:,0])).to_frame()
            if drifted:
                # store drifted weights if requested
                new_weights = curr_weights.reset_index()
                new_weights[DATE_NAME] = date
                new_weights = new_weights.set_index([DATE_NAME, STOCK_NAME])
                drifted_weights.append(new_weights)

            # normalize weights if requested
            if normalize and long_notional > 0:
#                curr_weights[curr_weights > 0] = portfolio_weights(curr_weights[curr_weights > 0], long_notional)
                curr_weights[curr_weights > 0] = (long_notional * curr_weights[curr_weights > 0] 
                                                    / curr_weights[curr_weights > 0].sum().abs().iloc[0])
            if normalize and short_notional > 0:
#                curr_weights[curr_weights < 0] = portfolio_weights(curr_weights[curr_weights < 0], short_notional)
                curr_weights[curr_weights < 0] = (short_notional * curr_weights[curr_weights < 0] 
                                                    / curr_weights[curr_weights < 0].sum().abs().iloc[0])
            
            # add drifted weights to portfolio if date was missing
            if date not in portfolio_dates:
                curr_weights = curr_weights.dropna().reset_index()
                curr_weights[DATE_NAME] = date
                curr_weights = curr_weights.set_index([DATE_NAME, STOCK_NAME])
                port_weights._frame = pd.concat([port_weights.frame, curr_weights], axis=0)

        # update previous weights
        prev_weights = port_weights.frame.xs(date, level=0).copy()

    # finally, sort the portfolio weights by date and stock
    port_weights._frame = port_weights._frame.sort_index(level=[0,1])
    if drifted: # return all drifted weights if requested
        return PanelFrame().set_frame(pd.concat(drifted_weights, axis=0).sort_index(level=[0,1]))
    else:       # only return imputed portfolio weights
        return port_weights


    # for date in tqdm(all_dates):
    #     if date in portfolio_dates:
    #         prev_weights = port_weights.frame.xs(date, level=0)
    #     else:
    #         if retx is not None and date in retx.frame.index.get_level_values(0):
    #             returns = retx.frame.xs(date, level=0).reindex(prev_weights.index, fill_value=0)
    #             curr_weights = (prev_weights.iloc[:,0] * (1 + returns.iloc[:,0])).to_frame()
    #             if normalize and long_notional > 0:
    #                 curr_weights[curr_weights > 0] = (long_notional * curr_weights[curr_weights > 0] 
    #                                                   / curr_weights[curr_weights > 0].sum().abs().iloc[0])
    #             if normalize and short_notional > 0:
    #                 curr_weights[curr_weights < 0] = (short_notional * curr_weights[curr_weights < 0] 
    #                                                   / curr_weights[curr_weights < 0].sum().abs().iloc[0])
    #         else:
    #             curr_weights = prev_weights.copy()
    #         prev_weights = curr_weights.copy()
    #         curr_weights = curr_weights.dropna().reset_index()
    #         curr_weights[DATE_NAME] = date
    #         curr_weights = curr_weights.set_index([DATE_NAME, STOCK_NAME])
    #         port_weights._frame = pd.concat([port_weights.frame, curr_weights], axis=0)
    # port_weights._frame = port_weights._frame.sort_index(level=[0,1])
    # return port_weights

    
if __name__ == "__main__":
    tic = time.time()

    # Get Universe
    dates = dict(start_date='1970-01-01', end_date='2024-12-31') #'2020-01-01'
    dates = dict(start_date='2020-01-01', end_date='2024-12-31') #
    nyse = PanelFrame('EXCHCD', **dates) == 1
    universe = PanelFrame('SIZE_DECILE', **dates)

    # Compute Book Value
    pstkrv = PanelFrame('pstkrv')
    pstkl = PanelFrame('pstkl')
    pstk = PanelFrame('pstk')
    seq = PanelFrame('seq')  # total shareholders' equity
#    preferred_stock = pstkrv\
#        .apply(replace, pstkl, how='outer', fill_value=0, values=[np.nan, 0])\
#        .apply(replace, pstk, how='outer', fill_value=0, values=[np.nan, 0])
    preferred_stock = pstkrv.fill(pstkl, pstk, replace=[0])
    txditc = PanelFrame('txditc').filter(end_date='1993-12-31')  # deferred taxes and investment tax credit

    book_value = seq - preferred_stock + txditc  # less preferred stock, add deferred tax before 1993

    age = seq.trend(cumcount)  # at least 2 years of age
    book_value = book_value.filter(min_value=0, dropna=True, mask=(age >= 2), **dates)

    # Compute Book to Market at December snapshots
    month = 12  # Decembers
    book_snapshot = factor_snapshots(book_value, month=month)
    print(f"Snapshot PanelFrame info: {frame_info(book_snapshot.frame)}")
    company_value = PanelFrame('CAPCO').filter(index=book_snapshot, min_value=1e-6, mask=universe>=0)

    # Lag Book to Market, restrict to universe and form terciles based on NYSE stocks
    lags = 6
    book_market = (book_snapshot.filter(index=company_value) / company_value).shift(lags).filter(min_stocks=100)

    # THIS SHOULD BE same as backtesting 

    bm_quantiles = book_market.apply(digitize, reference=nyse, cuts=[0.3, 0.7])

    # Restrict market value to universe, and form size quantiles based on NYSE stocks
    big_stocks = PanelFrame('SIZE_DECILE', **dates) <= 5   # big <=5, small >5

    # Form intersection
    market_value = PanelFrame('CAP', **dates)
#    small = bm_quantiles.filter(mask=(big_stocks == True)).apply(spread_portfolios, market_value)
#    big = bm_quantiles.filter(mask=(big_stocks == False)).apply(spread_portfolios, market_value)
#    composite_portfolio = (small + big) / 2  #portfolio_composite([small, big]) / 2
    BL = market_value.filter(mask=(big_stocks == True) & (bm_quantiles == 1)).apply(portfolio_weights)
    BH = market_value.filter(mask=(big_stocks == True) & (bm_quantiles == 3)).apply(portfolio_weights)
    SL = market_value.filter(mask=(big_stocks == False) & (bm_quantiles == 1)).apply(portfolio_weights)
    SH = market_value.filter(mask=(big_stocks == False) & (bm_quantiles == 3)).apply(portfolio_weights)
    composite_portfolio = (SH - SL + BH - BL) / 2

    drifted = portfolio_impute(composite_portfolio, PanelFrame('RETX'), drifted=True)
    turnover = composite_portfolio - drifted
    print(f"Average Turnover: {turnover.apply(np.abs).apply(np.sum, axis=0).apply(np.mean).frame}")

    composite_returns = portfolio_returns(composite_portfolio)
    summary = portfolio_evaluation(composite_returns)
    print(f"Composite Portfolio Summary: {summary}")

    bench = PanelFrame('HML').filter(index=composite_returns) / 100
    print(portfolio_evaluation(bench))
    composite_returns.plot(bench, kind='scatter', title='Composite BM vs HML')

    # sort by greatest difference between bench and composite returns
    diff = (composite_returns - bench).frame.sort_values(by=0, ascending=False)
    print(diff)
    toc = time.time()
    print(f"Total elapsed time: {toc - tic:.2f} seconds")
    raise Exception

    # Generate Book to Market Factor for all months with 6-month lag
    window = 11
    universe = 'ret_exc_lead1m'
    univ = PanelFrame(universe, **dates)
    bm = factor_generate(book_market, lags=lags, window=window, univ=univ)
    print(f"Generated Factor PanelFrame info: {frame_info(bm.frame)}")

    nyse = PanelFrame('crsp_exchcd') == 1
    size_quantiles = market_value.apply(digitize, reference=nyse, how='left', fill_value=False, cuts=2)
    bm_quantiles = bm.apply(terciles, reference=nyse, how='left', fill_value=False, cuts=[0.3, 0.7])

    small = bm_quantiles.filter(mask=(size_quantiles == 1)).apply(spread_portfolios, market_value)
    big = bm_quantiles.filter(mask=(size_quantiles == 2)).apply(spread_portfolios, market_value)

    composite_portfolio = big + small / 2
    composite_returns = port_returns(composite_portfolio)
    summary = portfolio_evaluation(composite_returns)
    print(f"Composite Portfolio Summary: {summary}")

    bench = PanelFrame('HML', **dates) / 100
    print(portfolio_evaluation(bench))
    composite_returns.plot(bench, kind='scatter', title='Composite BM vs HML')


if False:
    characteristic = 'book_equity'
    start_date = '2021-12-01'
    end_date = '2024-12-31'
    factor = PanelFrame(characteristic, start_date=start_date, end_date=end_date)
    print(f"Factor PanelFrame info: {frame_info(factor.frame)}")

    month = 12
    factor = factor_snapshots(factor, month=month)
    print(f"Snapshot PanelFrame info: {frame_info(factor.frame)}")

    window = 11
    lags = 6
    universe = 'ret_exc_lead1m'
    univ = PanelFrame(universe, start_date=start_date, end_date=end_date)
    factor = factor_generate(factor, lags=lags, window=window, univ=univ)
    print(f"Generated Factor PanelFrame info: {frame_info(factor.frame)}")

if False:
    print(str(datetime.now()))

    # Unit Test 6: pipeline
    ret = 'ret_vw_cap'
    factor = 'ret_12_1'
    dates = dict(start_date='1975-01-01', end_date='2024-12-31')
    dates = dict(start_date='2020-01-01', end_date='2024-12-31')

    factor_pf = PanelFrame(factor, **dates)

    size_pf = PanelFrame('me', **dates)

    nyse_pf = PanelFrame('crsp_exchcd', **dates).apply(pd.DataFrame.isin, values=[1, '1'])

    decile_pf = size_pf.apply(digitize, nyse_pf, cuts=10)

    quantiles_pf = factor_pf.apply(digitize, decile_pf > 1, cuts=3)

    vwcap_pf = PanelFrame('me', **dates).apply(winsorize, nyse_pf, lower=0, upper=0.80)

    spreads_pf = quantiles_pf.apply(spread_portfolios, vwcap_pf)

    lead1m_pf = PanelFrame('ret_exc_lead1m', **dates)
    facret_pf = (spreads_pf @ lead1m_pf).shift(shift=1)

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

    code = """
# Please run this following code:
from qrafti import PanelFrame, MEDIA
import matplotlib.pyplot as plt
ret = 'ret_vw_cap'
factor = 'ret_12_1'
bench = PanelFrame(factor + '_' + ret).to_frame()
bench.cumsum().plot()
savefig = MEDIA / f"{factor}_{ret}.png"
print(savefig)
#plt.savefig(savefig)
#print(f"[Image File](file:///{str(savefig)})")
"""

