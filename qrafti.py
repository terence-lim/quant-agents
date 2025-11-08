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
import markdown
from weasyprint import HTML

import pandas as pd
from pandas.api.types import is_list_like, is_integer_dtype, is_scalar, is_numeric_dtype, is_float_dtype
import matplotlib.pyplot as plt
import warnings

from portfolio import PortfolioEvaluation
pd.set_option('future.no_silent_downcasting', True) # for fillna behavior

DATA_LAKE = Path('/home/terence/Downloads/scratch/2024/JKP/')
WORKSPACE = DATA_LAKE / 'workspace'
MEDIA = DATA_LAKE / 'media'

STOCK_NAME = 'permno'
DATE_NAME = 'eom'
CALENDAR_PANEL = 'TOTAL_COUNT' # 'ret_exc_lead1m'

CRSP_VERSION = False
DATES = dict(start_date='2020-01-01', end_date='2024-12-31')

research_prompt = f"""You are a sell-side quantitative researcher writing a captivating research memo
on this new financial signal for predicting stock returns. You should also provide a title name for the signal.

Please follow these guidelines for writing the research memo:

1. Motivation (1 paragraph, ~100 words): 
    * Broad statement on market efficiency or asset pricing. 
    * Identify a gap in the current practice and literature.
    * Use active voice and declarative statements.

2. Hypothesis Development (1 paragraph, ~150 words):
    * Present economic mechanisms linking signal to returns.
    * Draw on theoretical frameworks.
    * Support claims with citations.

3. Results Summary (1-2 paragraphs, ~200 words):
    * Lead with the strongest statistical finding.
    * Summarize the key results in a narrative form, including economic significance.
    * Do not merely cite numbers; interpret them.

4. Contribution (1 paragraph, ~150 words):
    * Position relative to 3-4 related finance/accounting journal articles.
    * Highlight methodological innovations.

In your writing, please:

* Use active voice (e.g., “We find”).
* Maintain clarity and conciseness.
* Avoid jargon; explain technical terms.
* Use present tense for established findings.
* Use past tense for specific results.
* Make clear distinctions between correlation and causation.
* Avoid speculation beyond the data.

Output in markdown format with sections: Introduction, Hypothesis Development, Results, Contribution.

Base the results section strictly on the following data, matching its terminology and precision:
"""


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
# 1. JKP variables that are stems of factor-returns
# 2. all PSTAT variables
# 3. all CRSP variables with descriptions like JKP or my own
# 4. => RAG by bow: at least max(30, sqrt(len(variables)))
# should inject "Compustat Annual" for all pstat variables -> "Source: Compustat Annual" etc

def load_variables(filenames = ['PSTAT.csv', 'JKP.csv'], 
                   data_path: Path = DATA_LAKE) -> pd.DataFrame:
    """Read names, types and descriptions file"""
    keep_list = {'RET', 'RETX', 'PRC', 'VOL', 'SICCD', 'pstkrv','pstkl','pstk','seq','txditc','ret_12_1',
                 'HML', 'RF', 'Mkt-RF', 'RMW', 'CMA', 'SMB'}
    if CRSP_VERSION:
        keep_list |= {'CAPCO', 'CAP', 'SIZE_DECILE', 'SHRCD', 'EXCHCD', 'SIZE_DECILE'}
    else:  # should be different length, to trigger recreation
        keep_list |= {'me','me_company','size_grp','crsp_shrcd','crsp_exchcd'}

    try:
        df = pd.read_csv(data_path / 'characteristics.csv', index_col=0, sep='\t')
        assert len(df) == len(keep_list), "characteristics.csv does not have expected number of variables"
    except:
        suf = ' [Source: CRSP Monthly]'
        df = pd.DataFrame(index=['CAPCO', 'CAP', 'PRC', 'RET', 'RETX', 'VOL', 'SHRCD', 'EXCHCD', 'SICCD', 'SIZE_DECILE'],
                          data={'Type': ['float']*10,
                                'Description': ['Market Capitalization of Company' + suf,
                                                'Market Value of Common Equity' + suf,
                                                'Closing Stock Price' + suf,
                                                'Total Stock Return' + suf,
                                                'Stock Price Return without dividends' + suf,
                                                'Trading Volume' + suf,
                                                'Share Code ([10, 11]=Domestic US Common Stocks)' + suf,
                                                'Exchange Code (1=NYSE, 2=AMEX, 3=NASDAQ)' + suf,
                                                'Standard Industrial Classification Code' + suf,
                                                'Size Decile Classification (1=Largest, 10=Smallest)' + suf]})
        suf = ' [Source: Fama-French Research Factors]'
        bench = pd.DataFrame(index=['HML', 'RF', 'Mkt-RF', 'RMW', 'CMA', 'SMB'],
                             data={'Type': ['float']*6,
                                   'Description': ['HML Value benchmark returns' + suf,
                                                   'RF risk free rate' + suf,
                                                   'Market excess returns' + suf,
                                                   'RMW Quality benchmark returns' + suf,
                                                   'CMA Investment benchmark returns' + suf,
                                                   'SMB Size benchmark returns' + suf]})
                                                   
        df_list = [df, bench]
        for filename in filenames:
            df = pd.read_csv(data_path / filename, sep='\t', index_col=0, header=0)
            df.columns = ['Type', 'Description']
            # remove prefix substring between parentheses in Description
            df['Description'] = df['Description'].apply(lambda x: re.sub(r'^\(.*?\)\s*', '', x))
            if filename.lower().startswith('pstat'):
                df['Description'] = df['Description'] + ' (Source: Compustat Annual)'
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
        df.index.name = 'Name'
    
        # keep only rows with index value in keep_list
        print(keep_list)
        df = df[df.index.isin(keep_list)]
        df.to_csv(data_path / 'characteristics.csv', sep='\t')
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
    def __init__(self, start_date: str = '', end_date: str = '', reference_panel: str = CALENDAR_PANEL):
        # Initialize the Calendar with unique sorted dates from a reference Panel 'ret_exc_lead1m'
        dates = Panel(reference_panel).frame.index.get_level_values(0)
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
        dates_dict = {d[:4]+d[5:7]: d for d in self.dates.index}    # 'YYYYMM'
        dates_dict |= {int(d): date for d, date in dates_dict.items()}  # int YYYYMM
        dates_dict |= {int(d[:4]+d[5:7]+d[8:10]): d for d in self.dates.index}  # int YYYYMMDD
        dates_dict |= {d:d for d in self.dates.index}
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


###########################
#
# Data Structures to support tools for Research Agents
#
###########################

# Utility functions for Panels
def frame_info(frame: pd.DataFrame) -> Dict[str, Any]:
    """Return basic information about a DataFrame."""
    return frame.groupby(level=0).size().to_dict()

def panel_or_numeric(x: str, **kwargs) -> Union['Panel', float, int]:
    """Convert a string to a Panel or numeric value."""
    if x is None or x.lower() in ['', 'none']:
        return None
    try:
        if '.' in x:
            return float(x)
        else:
            return int(x)
    except:
        return Panel(x, **kwargs)

# TO DO: All date and vulnerable str arguments for Panel methods should be converted using str_or_None
def str_or_None(x: str) -> Union[str, None]:
    """Convert a string to None if it is 'None' or empty."""
    if x is None or x.lower() in ['', 'none']:
        return None
    return x

def numeric_or_None(x: str) -> Union[float, None]:
    """Convert a string to float or None if it is 'None' or empty."""
    try:
        return float(x)
    except:
        return None

def int_or_None(x: str) -> Union[float, None]:
    """Convert a string to float or None if it is 'None' or empty."""
    try:
        return int(float(x))
    except:
        return None

class Panel:
    """
    A Panel is a wrapper around a pandas DataFrame with multi-index (date, stock),
    representing a panel data structure commonly used in finance and econometrics.
    It supports various operations such as arithmetic, logical, grouping, and advanced
    operations on the panel data.
    """

    def __init__(self, name: str = '', start_date: str = DATES['start_date'], end_date: str = DATES['end_date']):
        """Initialize a Panel, optionally from a cached DataFrame file and date range.

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
        """Return the number of data items in this Panel."""
        return 0 if self._frame is None else len(self._frame)
    
    def __str__(self) -> str:
        return json.dumps({'results_panel_id': self.name}, indent=2)
    # return json.dumps({'results_panel_id': self.name, 'meta': self.info}, indent=2)

    @property
    def frame(self) -> pd.DataFrame:
        """Return the underlying DataFrame or scalar value of this Panel."""
        if self.nlevels == 0:
            return self._frame.iloc[0,0]
        return self._frame

    @property
    def info(self) -> Dict[str, Any]:
        """Return basic information about this Panel."""
        info = {'nlevels': self.nlevels, 'rows': len(self)}
        info['memory_usage_bytes'] = 0 if self.nlevels < 0 else int(self.frame.memory_usage(deep=True).sum())
        if self.nlevels >= 1:
            dates = self.dates
            info['num_dates'] = len(dates)
            info['min_date'] = str(dates[0])
            info['max_date'] = str(dates[-1])
        if self.nlevels == 2:
            info['max_stocks_per_date'] = int(self.frame.groupby(level=0).size().max())
            info['min_stocks_per_date'] = int(self.frame.groupby(level=0).size().min())
        return info

    @property
    def nlevels(self) -> int:
        """Number of index levels of this Panel
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
        """Return the values of this Panel as a numpy array.
        Returns:
            [] if empty, 1D array otherwise
        """
        return [] if self.nlevels < 0 else self.frame.iloc[:, 0].values
        
    @property
    def dates(self) -> List[str]:
        """Return the list of unique dates in this Panel."""
        return [] if self.nlevels <= 0 else sorted(self.frame.index.get_level_values(0).unique())

    def __getitem__(self, key: Union[str, int, Tuple]) -> Any:
        """Get item(s) from this Panel using indexing.
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
    def join_frame(self, other: 'Panel', fill_value: Any, how: str) -> pd.DataFrame:
        """Helper to join columns from another Panel, and return as a DataFrame
        Arguments:
            other: Another Panel to join with, or a scalar value to add as a column
            fill_value: Value to fill missing values in the other Panel
            how: Type of join to perform ('left', 'right', 'inner', 'outer')
        Returns:
            df: DataFrame with the joined data
        """
        assert (self.nlevels != 0) or (isinstance(other, Panel) and other.nlevels > 0), "At least one Panel with index levels"
        if self.nlevels > 0:
            # self is a multi-index Panel:
            df = self.frame.copy()
        else:
            # self is a scalar Panel or None: create DataFrame with same index as other
            data = fill_value if self._frame is None else self.frame   # value for self Panel
            df = pd.DataFrame(index=other.frame.index, data=data, columns=['self'])

        if isinstance(other, Panel):  # other is a Panel
            if other.nlevels > 0:
                # other is also a multi-index Panel: join on index levels
                other_df = other.frame
                assert df.index.nlevels == other_df.index.nlevels, "Cannot join Panels with different index levels"
                df = df.join(other_df, how=how, rsuffix='_').fillna(fill_value)
            elif other.nlevels == 0:
                # other is a scalar Panel: add as a column with same scalar value
                df['other'] = other.frame
            else:
                # other is None: add as column with fill-value
                df['other'] = fill_value
        elif other is None:
            df['other'] = fill_value
        else:
            df['other'] =  other
        if any(is_float_dtype(dtype) for dtype in df.dtypes):
            return df.astype("Float64")
        else:
            return df


    #
    # Primitive operations
    #
    def copy(self) -> 'Panel':
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
        
    def set_name(self, name: str) -> 'Panel':
        """Set the name of this Panel."""
        self.name = name
        if isinstance(self._frame, pd.DataFrame):
            self._frame.columns = [name]
        return self
    
    def set(self, value: int | float, index: 'Panel' = None) -> 'Panel':
        """Set values in this Panel at the specified index positions.
        Arguments:
            index: A Panel specifying the index positions to set (must have same index levels)
            value: The value(s) to set at the specified index positions (scalar or Panel)
        Returns:
            self: This Panel with updated values
        """
        if not index:
            self._frame = pd.DataFrame(value, index=[''], columns=[''])
            self._frame.index.name = None
        else:
            self._frame = pd.DataFrame(value, index=index.frame.index, columns=[''])
            self._frame.index.names = [DATE_NAME, STOCK_NAME][:index.nlevels]
        return self
    
    def set_frame(self, frame: pd.DataFrame, append=False) -> 'Panel':
        """Helper to set or append a DataFrame to this Panel."""

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

        # Convert input Series to DataFrame for processing
        if isinstance(frame, pd.Series):
            frame = frame.to_frame()

        if frame is None:    # None frame -> None
            self._frame = None
        elif is_scalar(frame):  # scalar value -> scalar-like DataFrame
            self._frame = _scalar_as_frame(frame)
        elif isinstance(frame, pd.DataFrame):
            if frame.empty:     # empty frame -> None
                self._frame = None
            elif _frame_is_scalar(frame):  # scalar-like DataFrame 
                self._frame = _scalar_as_frame(frame)
            elif append and self.frame is not None:   # append new values to existing
                old_frame = self.frame
                frame.columns = old_frame.columns # force new column names to match existing

                # drop duplicates based on index, keep the last occurrence
                self._frame = pd.concat([old_frame, frame], axis=0).sort_index(level=range(frame.index.nlevels))
                self._frame = self.frame[~self.frame.index.duplicated(keep='last')]
            else:
                self._frame = frame.sort_index(level=range(frame.index.nlevels))
                self._frame = self.frame[~self.frame.index.duplicated(keep='last')]
                self._frame.index.names = [DATE_NAME, STOCK_NAME][:frame.index.nlevels] # ensure index names
        else:
            assert False, "Frame must be a pandas DataFrame or scalar"
        return self

    def astype(self, dtype) -> 'Panel':
        """Change the dtype of the values of this Panel."""
        if isinstance(self._frame, pd.DataFrame):
            self._frame = self._frame.astype(dtype)
        return self

    def persist(self, name: str = '') -> 'Panel':
        """Set this Panel to persist its data to cache file.
        Arguments:
            name: Optional name for the file, if not given, a new name will be generated
        Returns:
            self: This Panel
        """
        name = DataCache.write_frame(frame=self._frame, name=name)
        self.name = name
        return self
        
    #
    # Panel Binary Operators
    #
    def _operands(self, other: 'Panel', fill_value: Any, how: str) -> Tuple[pd.Series, pd.Series]:
        """Internal helper to align another Panel or scalar to this Panel
        Notes: If other is a Panel, perform an outer join, and fill missing values with fill_value
        """
        if isinstance(other, Panel):
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

    def __add__(self, other: 'Panel') -> 'Panel':
        """Add values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=0, how='outer')
        return Panel().set_frame(df + df_other)

    def __radd__(self, other: 'Panel') -> 'Panel':
        """Add values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=0, how='outer')
        return Panel().set_frame(df_other + df)
    
    def __sub__(self, other: 'Panel') -> 'Panel':
        """Subtract values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=0, how='outer')
        return Panel().set_frame(df - df_other)

    def __rsub__(self, other: 'Panel') -> 'Panel':
        """Subtract values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=0, how='outer')
        return Panel().set_frame(df_other - df)
    
    def __mul__(self, other: 'Panel') -> 'Panel':
        """Multiply values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=1, how='inner')
        return Panel().set_frame(df * df_other)
    
    def __rmul__(self, other: 'Panel') -> 'Panel':
        """Multiply values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=1, how='inner')
        return Panel().set_frame(df_other * df)
    
    def __truediv__(self, other: 'Panel') -> 'Panel':
        """Divide values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=1, how='inner')
        return Panel().set_frame(df / df_other)
    
    def __rtruediv__(self, other: 'Panel') -> 'Panel':
        """Divide values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=1, how='inner')
        return Panel().set_frame(df_other / df)

    def __eq__(self, other: 'Panel') -> 'Panel':
        """Check equality of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how='inner')
        return Panel().set_frame((df == df_other).astype(bool))
    
    def __ge__(self, other: 'Panel') -> 'Panel':
        """Check equality of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how='inner')
        return Panel().set_frame((df >= df_other).astype(bool))

    def __gt__(self, other: 'Panel') -> 'Panel':
        """Check equality of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how='inner')
        return Panel().set_frame((df > df_other).astype(bool))

    def __le__(self, other: 'Panel') -> 'Panel':
        """Check equality of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how='inner')
        return Panel().set_frame((df <= df_other).astype(bool))

    def __lt__(self, other: 'Panel') -> 'Panel':
        """Check equality of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how='inner')
        return Panel().set_frame((df < df_other).astype(bool))

    def __ne__(self, other: 'Panel') -> 'Panel':
        """Check inequality of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=np.nan, how='inner')
        return Panel().set_frame((df != df_other).astype(bool))

    def __or__(self, other: 'Panel') -> 'Panel':
        """Logical or of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=0, how='outer')
        return Panel().set_frame((df.astype(bool) | df_other.astype(bool)))

    def __and__(self, other: 'Panel') -> 'Panel':
        """Logical or of values of this Panel with other"""
        df, df_other = self._operands(other, fill_value=1, how='inner')
        return Panel().set_frame((df.astype(bool) & df_other.astype(bool)))

    #
    # Panel Unary Operators
    #
    def __neg__(self) -> 'Panel':
        """Negate the values of this Panel."""
        return Panel().set_frame(-self.frame)
    
    def __invert__(self) -> 'Panel':
        """Boolean negation of the values of this Panel."""
        return Panel().set_frame(~self.frame.astype(bool))

    # TO DO: log and exp are not unary -- should be apply, pow and rpow are binary
    #def log(self) -> 'Panel':
    #    """Logarithm of the values of this Panel."""
    #    return Panel().set_frame(self.frame.apply(np.log))

    #def exp(self) -> 'Panel':
    #    """Exponentiate the values of this Panel."""
    #    return Panel().set_frame(self.frame.apply(np.exp))

    #def __pow__(self, other):
    #    # Allow exp-like behavior: e.g., math.e ** x
    #    return MyNumber(self.value ** other)

    #def __rpow__(self, other):
    #    # Allow exp(x) ≈ math.e ** x
    #    return MyNumber(other ** self.value)

    #
    #
    # Panel Group Utility Functions
    #
    def apply(self, func: Callable, reference: 'Panel' = None, fill_value=0, how='left', **kwargs) -> 'Panel':
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
            return Panel().set_frame(df)
        else:
            df.columns = cols[:len(df.columns)]
            return Panel().set_frame(pd.DataFrame(df).iloc[:, [0]])

    def trend(self, func: Callable) -> 'Panel':
        """Apply a function to compute a trend by stock over time"""
        assert self.nlevels == 2, "Trend can only be computed for Panels with 2 index levels"
        df = self.frame.reset_index().sort_values(by=[STOCK_NAME, DATE_NAME])
        df['trend'] = df.groupby([STOCK_NAME]).apply(func, include_groups=False).values
        df = df.set_index([DATE_NAME, STOCK_NAME]).sort_index()
        return Panel().set_frame(pd.DataFrame(df).iloc[:, [-1]])

    #
    # Panel Group Operations    
    #
    def __matmul__(self, other: 'Panel') -> 'Panel':
        """Compute the dot product of two Panels, by first index level date group."""
        def dot(x):
            """Dot product of two columns"""
            return (x.iloc[:, 0] * x.iloc[:, -1]).sum()
        return self.apply(dot, other, how='inner')

    #
    # Panel Advanced Operations
    #
    def shift(self, shift: int = 1) -> 'Panel':
        """Shift the dates of this Panel"""
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

    def filter(self, min_value: float = None, max_value: float = None,
               start_date: str = None, end_date: str = None, 
               dropna: bool = False, mask: 'Panel' = None, index: 'Panel' = None,
               min_stocks: int = None) -> 'Panel':
        """Filter the values of this Panel based on date, stock, and value criteria.
        Arguments:
            start_date: Optional start date to filter the DataFrame (inclusive)
            end_date: Optional end date to filter the DataFrame (inclusive)
            min_stocks: Optional minimum number of stocks per date to keep the date
            min_value: Optional minimum value to keep the row
            max_value: Optional maximum value to keep the row
            mask: Optional Panel of boolean values to filter the DataFrame
            index: Optional Panel whose index to keep in the DataFrame
            dropna: If True, drop rows with NaN values
        Returns:
            Panel with the filtered data
        """
        out_panel = self.copy()
        if self.nlevels < 1:   # empty or scalar
            return out_panel
        df = self._frame.copy()
        start_date = str_or_None(start_date)
        if start_date:
            df = df[df.index.get_level_values(0) >= start_date]
        end_date = str_or_None(end_date)
        if end_date:
            df = df[df.index.get_level_values(0) <= end_date]
        min_value = numeric_or_None(min_value)
        if min_value is not None:
            df = df[df.iloc[:, 0] >= min_value]
        max_value = numeric_or_None(max_value)
        if max_value is not None:
            df = df[df.iloc[:, 0] <= max_value]
        if dropna:
            df = df[df.iloc[:, 0].notna()]
        if isinstance(mask, Panel) and mask.nlevels == self.nlevels:
            mask_df = mask.frame
            if df.index.nlevels != mask_df.index.nlevels:
                raise ValueError("Cannot apply mask Panel with different index levels")
            df = df.join(mask_df, how='inner', rsuffix='_mask')
            df = df[df.iloc[:, -1].astype(bool)]  # keep only rows where mask is True
            df = df.iloc[:, :-1]  # drop the mask column
        if isinstance(index, Panel) and index.nlevels == self.nlevels:
            # only keep indexes that are in index.frame
            index_df = index.frame
            if df.index.nlevels != index_df.index.nlevels:
                raise ValueError("Cannot apply index Panel with different index levels")
            df = df.join(index_df, how='inner', rsuffix='_index')
            df = df.iloc[:, :-1]  # drop the index column
        min_stocks = numeric_or_None(min_stocks)
        if is_numeric_dtype(min_stocks) and self.nlevels == 2:
            counts = df.groupby(level=0).size()
            valid_dates = counts[counts >= min_stocks].index
            df = df[df.index.get_level_values(0).isin(valid_dates)]
        return out_panel.set_frame(df, append=False)

    def plot(self, other_panel: 'Panel' = None, **kwargs):
        """Plot the values of this Panel.
        Arguments:
            other_panel: Optional other Panel to plot on the same axes
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
# Common tools to be constructed with Panel.apply()
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

def portfolio_weights(x, leverage: float = 1.0, net: bool = True) -> pd.Series:
    """Scale the the portfolio weights to sum to the given leverage
    Arguments:
        x: DataFrame with at least two columns, first column is the raw portfolio weights,
           last column is a boolean indicator for which rows to consider for scaling
        leverage: Total leverage to scale the weights to
        net: If False, scale the average of the sum of absolute long and short weights to the leverage.  
             If True (default), scale the absolute sum of weights to the leverage. 
    Returns:
        pd.Series with the scaled weights
    Usage:
        panel_frame.apply(portfolio_weights, leverage=leverage, net=False)
    """
    # set weights to zero for rows where second column is False
    x.loc[~x.iloc[:, 1].astype(bool), x.columns[0]] = 0.0
    long_weight = x.loc[x.iloc[:,0] > 0, x.columns[0]].sum()
    short_weight = x.loc[x.iloc[:,0] < 0, x.columns[0]].sum()
    if net:
        total_weight = abs(long_weight + short_weight)
    else:
        total_weight = (abs(long_weight) + abs(short_weight)) / 2
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
# Common functions to be used with Panel.trend()
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
# Characteristics Functions
#
def characteristics_fill(*panels, replace: List = []) -> Panel:
    """Fill with values from other Panels in order

    Arguments:
        panels: Panels to use for filling values
    Returns:
        Panel with the filled values
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
        out_panel = out_panel.apply(replace_helper, panel, how='outer', fill_value=np.nan, replace=replace)
    return out_panel

def characteristics_downsample(characteristics: Panel, ffill: bool=True, month: List | int = []) -> Panel:
    """Downsamples characteristics values at lower frequency of a specific months
    Arguments:
        characteristics: Panel of characteristics values
        ffill (optional, bool): If True, forward fill values between months.  
          If False, only use values in the specified months.
        month: List of months (1-12) to extract downsamples for, if empty, extract for all months
    Returns:
        Panel of characteristics values at the specified date, forward filled from previous dates
    """
    assert characteristics.nlevels == 2, "characteristics must have two index levels"

    characteristics_dates = characteristics.dates
    prev_date = characteristics_dates[0]
    cal = Calendar(start_date=characteristics_dates[0], end_date=characteristics_dates[-1])
    samples_df = []
    for next_date in cal.dates_range(cal.start_date, cal.end_date):
        if not month or cal.ismonth(next_date, month):
            for curr_date in cal.dates_range(prev_date, next_date):
                if curr_date in characteristics_dates:
                    if ffill or next_date == curr_date:
                        characteristics_df = characteristics.frame.xs(curr_date, level=0).reset_index()
                        characteristics_df[DATE_NAME] = next_date
                        characteristics_df['_date_'] = curr_date
                        samples_df.append(characteristics_df)
            prev_date = cal.offset(next_date, 1, strict=True)

    # sort by STOCK_NAME, DATE_NAME and _date_ and drop duplicates, keep last
    samples_final = pd.concat(samples_df, axis=0)
    samples_final = samples_final.sort_values(by=[STOCK_NAME, DATE_NAME, '_date_'])
    samples_final = samples_final.drop_duplicates(subset=[STOCK_NAME, DATE_NAME], keep='last')
    samples_final = samples_final.set_index([DATE_NAME, STOCK_NAME]).drop(columns=['_date_'])
    samples_panel = Panel().set_frame(samples_final)
    return samples_panel


#
# Portfolio Functions
#
def portfolio_impute(port_weights: Panel, retx: Panel = None,
                     normalize: bool = True, drifted: bool = False) -> Panel:
    """Impute missing portfolio weights on missing dates by forward drifting previous weights.
    Arguments:
        port_weights: Panel of portfolio weights
        retx: Panel of stock returns to forward drift previous weights
        normalize: If True, re-normalize weights to be dollar-neutral after forward drifting
        drifted: Whether to output all drifted weights (True), or only fill in missing dates
    Returns:
        Panel of portfolio weights with missing dates imputed by forward drifting
    Notes:
        Side effect: Changes port_weights in place where missing dates are added.
    """
    assert port_weights.nlevels == 2, "Portfolio weights must have two index levels"
    if retx is None:
        if CRSP_VERSION:
            retx = Panel('RETX')   # RETX should be actual dates, for drifting previous weights only
        else:
            retx = Panel('ret_exc_lead1m').shift(1)
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
    for date in tqdm(all_dates, desc='portfolio_impute'):
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
                curr_weights[curr_weights > 0] = (long_notional * curr_weights[curr_weights > 0] 
                                                    / curr_weights[curr_weights > 0].abs().sum().iloc[0])
            if normalize and short_notional > 0:
                curr_weights[curr_weights < 0] = (short_notional * curr_weights[curr_weights < 0] 
                                                    / curr_weights[curr_weights < 0].abs().sum().iloc[0])
            
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
        return Panel().set_frame(pd.concat(drifted_weights, axis=0).sort_index(level=[0,1]))
    else:       # only return imputed portfolio weights
        return port_weights

def portfolio_returns(port_weights: 'Panel', price_changes: 'Panel' = None,
                      stock_returns: 'Panel' = None) -> 'Panel':
    """Compute the portfolio returns given portfolio weights and stock returns.
    Arguments:
        port_weights: Panel of portfolio weights
        stock_returns: Panel of leading stock returns
    Returns:
        Panel of portfolio returns, shifted by one date to align with end of holding period
    """
    if stock_returns is None:
        if CRSP_VERSION:
            stock_returns = Panel('EXCRET').shift(-1)  # should be leading dates, to compute realized returns
        else:
            stock_returns = Panel('ret_exc_lead1m')

    port_weights = portfolio_impute(port_weights, retx=price_changes, normalize=True)
    return (port_weights @ stock_returns).shift(1)

def portfolio_metrics(port_returns: Panel) -> Dict[str, float]:
    """Compute summary performance statistics of a portfolio given its weights and returns.
    Arguments:
        port_returns: Panel of portfolio returns
    Returns:
        Dict of summary statistics: mean return, volatility, Sharpe ratio
    """
    return {} if port_returns.nlevels != 1 else PortfolioEvaluation(returns=port_returns.frame).metrics()

def portfolio_regression(port_returns: Panel, factor_returns: List[Panel] = []) -> Dict[str, float]:
    """Compute regression coefficients of a portfolio given its returns and factor returns.
    Arguments:
        port_returns: Panel of portfolio returns
        factor_returns: List of Panel of factor returns
    Returns:
        Dict of coefficients and statistics from regression of portfolio returns on intercept and factor returns
    """
    if port_returns.nlevels != 1:
        return {}
    factor_frames = [factor.frame for factor in factor_returns if factor.nlevels == 1]
    return PortfolioEvaluation(returns=port_returns.frame).regression(factor_frames)

def factor_evaluate(signal: Panel) -> str:
    """compute factor returns, evaluation statistics, regression analysis.
    Arguments:
        signal: Panel of signal values from which to calculate and evaluate factor returns
    Returns:
        str: Evaluation results and tables in markdown format
    Notes (TO DO): 
        @mcp.tool 
        def Panel_factor_evaluate(factor: str) -> str
        # Creates the prompt and context to respond with the factor evaluation
    """       
    context = []

    # Coverage of stocks
    total_count = Panel('TOTAL_COUNT').filter(index=signal)
    signal = signal.filter(index=total_count).filter(index=total_count)
    count_coverage = (signal.filter(index=total_count).apply(len)/total_count).set_name('count')

    # coverage by year
    df = count_coverage.frame.reset_index()
    df['year'] = df[DATE_NAME].str.slice(0,4).astype(int)
    yearly_count = df.groupby('year')['count'].mean().to_frame(name='% of names covered') * 100
    context.append('### % of Names Covered by Year\n' + yearly_count.round(2).to_markdown())

    # Coverage of cap
    total_cap = Panel('TOTAL_CAP').filter(index=signal)
    cap = Panel('CAP').filter(index=signal)
    cap_coverage = (cap.apply(pd.DataFrame.sum)/total_cap).set_name('cap')
    df = cap_coverage.frame.reset_index()
    df['year'] = df[DATE_NAME].str.slice(0,4).astype(int)
    yearly_cap = df.groupby('year')['cap'].mean().to_frame(name='% of cap covered') * 100
    context.append('### % of Market Cap Covered by Year\n' + yearly_cap.round(2).to_markdown())

    # Form portfolios
    quantiles = signal.apply(digitize, fill_value=True, cuts=3)
    capvw = Panel('CAPVW').filter(index=signal)
    q3 = capvw.apply(portfolio_weights, reference=quantiles == 3, how='right')
    q1 = capvw.apply(portfolio_weights, reference=quantiles == 1, how='right')
    portfolio = q3 - q1

    # turnover
    #drifted = portfolio_impute(portfolio, drifted=True)
    #trades = portfolio.filter(index=drifted) - drifted
    #turnover = trades.apply(pd.DataFrame.abs).apply(pd.DataFrame.sum)/2

    # Evaluate returns
    returns = portfolio_returns(portfolio)
    stats = portfolio_metrics(returns)
    df = pd.Series(stats, name='high-low tercile').to_frame().round(4).T
    context.append("### Statistics of Tercile Spread Portfolios\n(weighted by market cap winsorized at 80th NYSE percentile")
    context.append(df.round(4).to_markdown())
 
    # by model
    context.append("### Alpha, coefficients and t-statistics by Model") 
    mu = portfolio_regression(returns, [])
    df = pd.DataFrame({'coefficients': {'intercept': mu['intercept']} | mu['coefficients'],
                       't-stats': {'intercept': mu['t_intercept']} | mu['t_statistics']}
                      ).rename_axis(index='Mean Returns')
    context.append(df.round(4).to_markdown())

    capm = portfolio_regression(returns, [Panel('Mkt-RF')])
    df = pd.DataFrame({'coefficients': {'intercept': capm['intercept']} | capm['coefficients'],
                       't-stats': {'intercept': capm['t_intercept']} | capm['t_statistics']}
                      ).rename_axis(index='CAPM')
    context.append(df.round(4).to_markdown())

    ff3 = portfolio_regression(returns, [Panel('Mkt-RF'), Panel('SMB'), Panel('HML')])
    df = pd.DataFrame({'coefficients': {'intercept': ff3['intercept']} | ff3['coefficients'],
                       't-stats': {'intercept': ff3['t_intercept']} | ff3['t_statistics']}
                      ).rename_axis(index='Fama-French 3-Factor Model')
    context.append(df.round(4).to_markdown())

    # Evaluate alphas by size quintile
    size_decile = Panel('SIZE_DECILE').filter(index=signal)
    out = []
    for quintile, sz in enumerate([[1,2], [3,4], [5,6], [7,8], [9,10]]):
        size_mask = size_decile.apply(pd.DataFrame.isin, values=sz)
        quantiles_sz = signal.apply(digitize, size_mask, cuts=3)
        high_sz = Panel().set(1, index=quantiles_sz).apply(portfolio_weights, reference=(quantiles_sz == 3), how='right')
        low_sz = Panel().set(1, index=quantiles_sz).apply(portfolio_weights, reference=(quantiles_sz == 1), how='right')
        portfolio_sz = high_sz - low_sz
        returns_sz = portfolio_returns(portfolio_sz)
        mu_sz = portfolio_regression(returns_sz)
        capm_sz = portfolio_regression(returns_sz, [Panel('Mkt-RF')])
        ff3_sz = portfolio_regression(returns_sz, [Panel('Mkt-RF'), Panel('SMB'), Panel('HML')])
        out.append(pd.DataFrame([mu_sz['intercept'], mu_sz['t_intercept'], 
                                 capm_sz['intercept'], capm_sz['t_intercept'], 
                                 ff3_sz['intercept'],  ff3_sz['t_intercept']],
                                index=['mean', 't-stat', 
                                       'alpha (CAPM)', 't-stat (CAPM)', 
                                       'alpha (FF3)', 't-stat (FF3)'],
                                columns=[f'Size Quintile {quintile+1}']))
    df = pd.concat(out, axis=1).rename_axis(index='Model')
    context.append("### Alpha and t-statistics by Model and Size Quintile\n(lower quintiles have smaller market cap)")
    context.append(df.round(4).to_markdown())

    context = "\n\n".join(context)
    return context


def markdown_to_pdf(markdown_text: str, stylesheets: List[str] = ['style.css'], 
                    output_file: str = 'output.pdf', debug: bool = False) -> Dict[str, str]:
    """Convert markdown text to PDF using WeasyPrint.
    Arguments:
        markdown_text: Markdown formatted string
        stylesheets: List of CSS stylesheet files to apply
        output_file: Output PDF file name
        debug: If True, print debug information
    """
    import markdown
    from weasyprint import HTML
    html_content = markdown.markdown(markdown_text, extensions=['tables'])
    if debug:
        print(html_content)
    html_doc = HTML(string=html_content)
    html_doc.write_pdf(output_file, stylesheets=stylesheets)
    return dict(output_file=output_file)

if __name__ == "__main__":
    print(load_variables())
    raise Exception

    def show(x):
        if isinstance(x, int):
            x = Panel(f'_{x}')
        elif isinstance(x, str):
            if not x.startswith('_') and x.isdigit():
                x = '_' + x
            x = Panel(x)
        print(x.frame)
        print(str(x))

    def p(x):
        if isinstance(x, int):
            return Panel(f'_{x}')
        elif isinstance(x, str):
            if not x.startswith('_') and x.isdigit():
                x = '_' + x
            return Panel(x)
        else:
            raise ValueError("Input must be int or str")
    tic = time.time()
    print(str(datetime.now()))

    dates = DATES
    dates = dict(start_date='1975-01-01', end_date='2024-12-31')
    dates = dict(start_date='2020-01-01', end_date='2024-12-31')




#    act = Panel('act', **dates)
#    ebitda = Panel('ebitda', **dates)
#    margin = act / ebitda

#    signal = Panel('CAP', **dates)

    # if False:    # DO NOT DELETE -- FF
    # Get Universe
#    universe = Panel('SIZE_DECILE')

    dates = dict(start_date='1970-01-01', end_date='2024-12-31') #'2020-01-01'
    dates = dict(start_date='2020-01-01', end_date='2024-12-31') #
    _dates = {}   # require all dates in case of aging
    _dates = dates

    # Compute Book Value
    pstkrv = Panel('pstkrv', **_dates)
    pstkl = Panel('pstkl', **_dates)
    pstk = Panel('pstk', **_dates)
    seq = Panel('seq', **_dates)  # total shareholders' equity
    preferred_stock = characteristics_fill(pstkrv, pstkl, pstk, replace=0)
#    preferred_stock = pstkrv | pstkl | pstk
    txditc = Panel('txditc', **_dates).filter(end_date='1993-12-31')  # deferred taxes and investment tax credit

    book_value = seq - preferred_stock + txditc  # less preferred stock, add deferred tax before 1993

#    age = seq.trend(cumcount)  # at least 2 years of age
#    book_value = book_value.filter(min_value=0, dropna=True, mask=(age >= 2), **dates)

    # Compute Book to Market at December samples
    month = 12  # Decembers
    book_samples = characteristics_downsample(book_value, month=month)
    print(f"Downsampled Panel info: {frame_info(book_samples.frame)}")
    company_value = Panel('CAPCO', **dates)
#    company_value = company_value.filter(index=universe)
    company_value = characteristics_downsample(company_value, month=month)

    # Lag Book to Market, restrict to universe and form terciles based on NYSE stocks
    lags = 6
    book_market = (book_samples / company_value).shift(lags)
#    book_market = book_market.filter(index=universe)

    # Form size and bm quantiles based on NYSE stocks
    nyse = (Panel('EXCHCD', **dates) == 1)
#    nyse = nyse.filter(index=universe)
    bm_quantiles = book_market.apply(digitize, reference=nyse, cuts=[0.3, 0.7])
    size_quantiles = Panel('CAP', **dates)
#    size_quantiles = size_quantiles.filter(index=universe)
    size_quantiles = size_quantiles.apply(digitize, nyse, cuts=2)
#    size_quantiles = (Panel('SIZE_DECILE') > 5) + 1

    # Form intersection
    market_value = Panel('CAP', **dates)
    BL = market_value.apply(portfolio_weights, reference=(size_quantiles==2) & (bm_quantiles == 1), how='right')
    BH = market_value.apply(portfolio_weights, reference=(size_quantiles==2) & (bm_quantiles == 3), how='right')
    SL = market_value.apply(portfolio_weights, reference=(size_quantiles==1) & (bm_quantiles == 1), how='right')
    SH = market_value.apply(portfolio_weights, reference=(size_quantiles==1) & (bm_quantiles == 3), how='right')
    composite_portfolio = (SH - SL + BH - BL) / 2

#    drifted = portfolio_impute(composite_portfolio, drifted=True)
#    turnover = composite_portfolio - drifted
#    print(f"Average Turnover: {turnover.apply(np.abs).apply(np.sum, axis=0).apply(np.mean).frame}")

    composite_returns = portfolio_returns(composite_portfolio)
    summary = portfolio_metrics(composite_returns)
    print(f"Composite Portfolio Summary: {summary}")

#    composite_returns = composite_returns.filter(start_date='1972-01-01')
    bench = Panel('HML', **dates).filter(index=composite_returns)
    print(portfolio_metrics(bench))
    composite_returns.plot(bench, kind='scatter', title='Composite BM vs HML')

    # sort by greatest difference between bench and composite returns
    diff = (composite_returns - bench).frame.sort_values(by=0, ascending=False)
    print(diff)
    print(f'Mean absolute difference: {diff.abs().mean().item():.4f}, '
          f'{np.corrcoef(composite_returns.frame.values, bench.frame.values, rowvar=False)[0,1].item():.4f}, '
          f'{composite_returns.frame.abs().mean().item():.4f}, {bench.frame.abs().mean().item():.4f}')

    toc = time.time()
    print(f"Total elapsed time: {toc - tic:.2f} seconds")
    raise Exception


if False:  # factor evaluate
    signal = Panel('ret_12_1', **dates)

    description = "12-1 Momentum: Prior 12 month returns excluding most recent month"
    context = "\n\n".join([description+'\n------------', research_prompt, factor_evaluate(signal)] )
    with open('output.md', 'w') as f:
        f.write(context)
    markdown_to_pdf(context)

    # Sanity plots
    returns.apply(pd.DataFrame.cumsum).plot(kind='line')
    (-Panel('SMB', **dates)).apply(pd.DataFrame.cumsum).plot(kind='line')
    returns.plot(-Panel('SMB', **dates), kind='scatter')
    plt.show()
    raise Exception

if False:  # JKP
    print(str(datetime.now()))

    # Unit Test 6: pipeline
    ret = 'ret_vw_cap'
    factor = 'ret_12_1'
    dates = dict(start_date='1975-01-01', end_date='2024-12-31')
    dates = dict(start_date='2020-01-01', end_date='2024-12-31')

    factor_pf = Panel(factor, **dates)

    size_pf = Panel('me', **dates)

    nyse_pf = Panel('crsp_exchcd', **dates).apply(pd.DataFrame.isin, values=[1, '1'])

    decile_pf = size_pf.apply(digitize, nyse_pf, cuts=10)

    quantiles_pf = factor_pf.apply(digitize, decile_pf > 1, cuts=3)

    vwcap_pf = Panel('me', **dates).apply(winsorize, nyse_pf, lower=0, upper=0.80)

    spreads_pf = quantiles_pf.apply(spread_portfolios, vwcap_pf)

    lead1m_pf = Panel('ret_exc_lead1m', **dates)
    facret_pf = (spreads_pf @ lead1m_pf).shift(shift=1)

    bench = Panel(factor + '_' + ret, **dates)
    facret_pf.plot(bench, kind='scatter')

    print(facret_pf)
    print(str(datetime.now()))

    panel_id = 'me'
    other_panel_id = 'ret_exc_lead1m'
    code = f"""
import json
from qrafti import Panel
dates = dict(start_date='2020-01-01', end_date='2024-12-31')
p1, p2 = Panel('{panel_id}', **dates), Panel('{other_panel_id}', **dates)
p3 = (p1 @ p2).persist()
print(json.dumps({{'result_panel_id': p3.name}}))
"""
    stdout, stderr, exit_code = run_code_in_subprocess(code)
    print(exit_code)
    print(stdout)

    code = """
# Please run this following code:
from qrafti import Panel, MEDIA
import matplotlib.pyplot as plt
ret = 'ret_vw_cap'
factor = 'ret_12_1'
bench = Panel(factor + '_' + ret).to_frame()
bench.cumsum().plot()
savefig = MEDIA / f"{factor}_{ret}.png"
print(savefig)
#plt.savefig(savefig)
#print(f"[Image File](file:///{str(savefig)})")
"""

