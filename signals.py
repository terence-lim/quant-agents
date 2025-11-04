# signals.py (Terence Lim, 2025)
import pandas as pd
import numpy as np
from pandas.api.types import is_list_like, is_integer_dtype
import matplotlib.pyplot as plt
from pathlib import Path
import re
from tqdm import tqdm
import json
from typing import List, Dict, Union, Set, Any, Tuple, Callable
from datetime import datetime
from pprint import pprint


import pandas as pd
import matplotlib.pyplot as plt
import warnings

from qrafti import STOCK_NAME, DATE_NAME, Panel, DATA_LAKE, Calendar, cumcount

PSTAT_DATA = Path('/home/terence/Downloads/scratch/2024/PSTAT')
CRSP_DATA = Path('/home/terence/Downloads/scratch/2024/CRSP')
BENCH_DATA = DATA_LAKE / 'factor-returns'

class Lookup:
    def __init__(self, source: str = 'gvkey', sep='\t'):
        """Initialize a Lookup object for mapping identifiers over time
        Args:
            filename (str | Path): Path to the lookup CSV file
            source (str): Column name for source identifiers
        Notes:
            # TODO: Handle dtypes of source and targe!
        """
        if source == 'gvkey':
#            filename = DATA_LAKE / 'ccm.txt.gz'
            filename = PSTAT_DATA / 'links.txt.gz'
            target = 'LPERMNO'
            date = 'LINKDT'
        if source == 'PERMNO':
            filename = CRSP_DATA / 'names.txt.gz'
            target = 'PERMCO'
            date = 'DATE'

        df = pd.read_csv(filename, sep=sep, header=0, low_memory=False).convert_dtypes()
        if is_integer_dtype(df[date]):
            df[date] = pd\
                .to_datetime(df[date], format='%Y%m%d', errors='coerce')\
                .dt.to_period('M')\
                .dt.to_timestamp('M')\
                .dt.strftime('%Y-%m-%d')
        df = df.sort_values([source, date]).dropna(subset=[source, target, date]).convert_dtypes()
        try:
            df = df.loc[df[source] > 0]
        except:
            df = df.loc[df[source].str.len() > 0]
        self.lookups = df.groupby(source)
        self.keys = set(self.lookups.indices.keys())
        self.source = source
        self.target = target
        self.date = date

    def __call__(self, stock: str, date: str = '2099-12-31', target: str = None) -> Any:
        """Return target identifiers matched to source as of date"""
        target = target or self.target
        if stock in self.keys:
            a = self.lookups.get_group(stock)
            b = a[a[self.date] <= date].sort_values(self.date)
#            return int((b.iloc[-1] if len(b) else a.iloc[0]).at[target])
            return b.iloc[-1].at[target] if len(b) else 0
        else:
            return 0


###########################
#
# load_pivot: for JKP benchmarks
#
###########################

def load_pivot_csv(values: str, 
                   index: str, 
                   columns: str,
                   sep: str,
                   filename: str | Path,
                   keep: List = []) -> pd.DataFrame:
    """Read benchmark returns from CSV file"""
    df = pd.read_csv(filename, sep=sep, header=0, low_memory=False)
    df = df.pivot(index=index, columns=columns, values=values).sort_index()
    if keep:
        df = df[keep]
    df = df.convert_dtypes()
    df.index.name = DATE_NAME
    return df

### To Load benchmarks
ret_type = 'ret_vw_cap'
# benchmarks_df = load_pivot_csv(values=ret_type, index='eom', columns='characteristic', sep=',',
#                                filename=BENCH_DATA / 'USA.csv')
# for bench in benchmarks_df.columns:
#      df = benchmarks_df[[bench]].dropna()
#      col = bench + '_' + ret_type
#      df.columns = [col]
#      Panel(col).set_frame(df).persist(col)

# TO DO:
# 1. names of JKP benchmark returns for "benchmarks.txt" -- then into load_variables ?
# 2. only keep those columns from load_pivot for "variables.txt"

###########################
#
# load_panel_csv: for PSTAT and JKP characteristics
#
###########################

def load_panel_csv(filename: str | Path, 
                   stock_name: str,
                   date_name: str,
                   sep: str,
                   append: bool,
                   aged: int,
                   filter: Dict ={},
                   keep: List[str] = []) -> None:
    """Load each column of CSV file to its own Panel."""
#if True:
#    stock_name = 'gvkey'
#    date_name = 'datadate'
#    sep = '\t'
#    filename = PSTAT_DATA / 'annual2010.txt.gz'
#    keep = ['seq', 'pstk', 'pstkrv', 'pstkl']

    df_data = pd.read_csv(filename, sep=sep, header=0, low_memory=False)
    if not keep:
        keep = list(set(df_data.columns) - {stock_name, date_name})

    # Convert dates
    cal = Calendar()
    df_data[date_name] = cal.as_dates(df_data[date_name])
    dates = df_data[date_name].unique()
    print('Dates not in calendar:', sorted(set(dates) - set(cal.dates.index)))

    # Apply filters
    df_data = df_data[df_data[date_name].isin(cal.dates.index)]
    for k,v in filter.items():
        if k in df_data.columns:
            df_data = df_data[df_data[k] == v]

    # Lookup stock identifiers by link date
    lookup = Lookup(source=stock_name)
    df_data[stock_name] = [lookup(s, d) for s,d in zip(df_data[stock_name].tolist(), 
                                                       df_data[date_name].tolist())]
    df_data = df_data[df_data[stock_name] > 0]
    df_data.set_index([date_name, stock_name], inplace=True)
    df_data.index.names = (DATE_NAME, STOCK_NAME)

    for i, col in tqdm(enumerate(keep), total=len(keep)):
        df = df_data[[col]].dropna().convert_dtypes()
        panel = Panel(col)
        print(panel.name, len(panel), panel.nlevels)
        panel = panel.set_frame(df, append=append)
        print('  - append:', panel.name, len(panel), panel.nlevels)
        if aged:
            age = panel.trend(cumcount)  # observation age
            panel = panel.filter(mask=(age >= aged)).persist(col)
            print(f'  - aged{aged}:', panel.name, len(panel), panel.nlevels)
        pprint(panel.info)
        #print(df.head())

#################
#
# load_csv - FF factor returns
#
#################
def load_csv(filename: str | Path, 
             date_name: str,
             sep: str,
             append: bool,
             mul: float = 1.0,
             keep: List[str] = []) -> None:
    """Load each column of CSV file to its own Panel."""
# if True:
#     filename = DATA_LAKE / 'FF.csv'
#     date_name = 'Date'
#     sep = '\t'
#     keep=['Mkt-RF', 'HML']

    df_data = pd.read_csv(filename, sep=sep, header=0, low_memory=False)
    if not keep:
        keep = list(set(df_data.columns) - {date_name})

    ### TO DO: Handle dates not in calendar, esp different month end dates
    cal = Calendar()
    df_data[date_name] = cal.as_dates(df_data[date_name])
    dates = df_data[date_name].unique()
    print('Dates not in calendar:', sorted(set(dates) - set(cal.dates.index)))
    df_data = df_data[df_data[date_name].isin(cal.dates.index)]

    for i, col in tqdm(enumerate(keep), total=len(keep)):
        df = df_data[[date_name, col]].dropna().convert_dtypes()
        df.set_index(date_name, inplace=True)
        df.index.name = DATE_NAME
        panel = Panel(col)
        print(panel.name, len(panel), panel.nlevels)
        panel = panel.set_frame(df * mul, append=append).persist(col)
        print(panel.name, len(panel), panel.nlevels)
        pprint(panel.info)
        #print(df.head())

# To load Fama-French factors
# load_csv(DATA_LAKE / 'FF.csv', mul=0.01, date_name='Date', sep='\t', keep=['Mkt-RF', 'HML'])

######################
#
# load_crsp - for CRSP Monthly
#
######################
def load_crsp(filename: str | Path, 
              date: str,
              sep: str,
              restrict_universe=False,
              keep: List[str] = []) -> None:
    """Load each column of CSV file to its own Panel."""

#    filename = DATA_LAKE / 'crsp.txt.gz'
#    date = 'date'
#    sep = '\t'
    df_data = pd.read_csv(filename, sep=sep, header=0, low_memory=False)
    df_data[date] = pd\
        .to_datetime(df_data[date], format='%Y%m%d', errors='coerce')\
        .dt.to_period('M')\
        .dt.to_timestamp('M')\
        .dt.strftime('%Y-%m-%d')
    df_data['DLSTCD'] = pd.to_numeric(df_data['DLSTCD'], errors='coerce').fillna(0).astype(int)
    df_data['DLRET'] = pd.to_numeric(df_data['DLRET'], errors='coerce').astype(float)
    df_data['PRC'] = pd.to_numeric(df_data['PRC'], errors='coerce').astype(float)
    df_data['RET'] = pd.to_numeric(df_data['RET'], errors='coerce').astype(float)
    df_data['RETX'] = pd.to_numeric(df_data['RETX'], errors='coerce').astype(float)
    df_data['VOL'] = pd.to_numeric(df_data['VOL'], errors='coerce').astype(float)
    df_data['SHROUT'] = pd.to_numeric(df_data['SHROUT'], errors='coerce').astype(float)
    df_data['CAP'] = df_data['SHROUT'] * df_data['PRC'].abs() / 1000
    df_data['SHRCD'] = pd.to_numeric(df_data['SHRCD'], errors='coerce').fillna(0).astype(int)
    df_data['EXCHCD'] = pd.to_numeric(df_data['EXCHCD'], errors='coerce').fillna(0).astype(int)
    df_data['PERMCO'] = pd.to_numeric(df_data['PERMCO'], errors='coerce').fillna(0).astype(int)
    df_data['SICCD'] = pd.to_numeric(df_data['SICCD'], errors='coerce').fillna(0).astype(int)

    # sum up CAP by date and PERMCO, then merge back to df_data
    cap = df_data.groupby([date, 'PERMCO'])['CAP'].sum().rename('CAPCO')
    df_data = df_data.merge(cap, on=[date, 'PERMCO'], how='left')

    # replace DLRET with -0.3 for these DLSTCD values when DLRET is missing (5129625->5129516)
    dlstcodes_ = set([500, 520, 580, 584]).union(list(range(551,575))) 
    replace_ = (df_data['DLRET'].isna() & df_data['DLSTCD'].isin(dlstcodes_))
    df_data.loc[replace_, 'DLRET'] = -0.3

    # restrict to domestic common stocks only
    universe_ = (df_data['SHRCD'].isin([10,11]) & 
                 df_data['EXCHCD'].isin([1,2,3]) &
                 (df_data['CAPCO'] > 1e-8) &
                 (df_data['CAP'] > 1e-8) &
                 (df_data['PRC'].abs() > 1e-8))
    if restrict_universe:
        length = len(df_data)
        df_data = df_data.loc[universe_, :]
        print('Restricted Universe:', length, len(df_data))

    # adjust RET and RETX for delisting returns
    replace_ = df_data['DLRET'].notna()
    df_data.loc[replace_, 'RETX'] =  (1+df_data['RETX'].fillna(0)) * (1+df_data['DLRET']) - 1
    df_data.loc[replace_, 'RET'] =  (1+df_data['RET'].fillna(0)) * (1+df_data['DLRET']) - 1

    for col in tqdm(['CAPCO', 'CAP', 'PRC', 'RET', 'RETX', 'VOL', 'SHRCD', 'EXCHCD', 'SICCD']):
        df = df_data[[date, 'PERMNO', col]]
        if col in ['CAPCO', 'CAP']:  # greater than 0, notna
            df = df[df[col] > 1e-4].dropna()
        if col in ['PRC']:  # not 0, not na
            df = df[df[col].abs() > 1e-4].dropna()
        if col in ['RET', 'RETX', 'VOL']:  # not na
            df = df.dropna()
        if col in ['SHRCD', 'EXCHCD', 'SICCD']:  # greater than 0, notna
            df = df[df[col] > 0].dropna()
        df.set_index([date, 'PERMNO'], inplace=True)
        df.index.names = (DATE_NAME, STOCK_NAME)
        panel = Panel(col).set_frame(df, append=False).persist(col)
        print(panel.name, len(panel), panel.nlevels)

    # Identify universe of stocks, and assign into size deciles based on stocks with exchcd == 1 (NYSE)
    def deciles(x: pd.DataFrame) -> pd.DataFrame:
        """Assign x['CAPCO'] into descending decile ranks using break points where x['EXCHCD'] is 1 (NYSE)"""
        #x = x.sort_values('CAPCO', ascending=True)
        nyse_ = x['EXCHCD'] == 1
        breakpoints = x.loc[nyse_, 'CAPCO'].quantile(np.linspace(0, 1, 11)).values
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        #print(len(x), x.iloc[0], breakpoints)
        ranks = pd.cut(x['CAPCO'], bins=breakpoints, labels=range(10,0,-1), include_lowest=True)
        return ranks.astype(int)   # decile ranks from 1 (largest) to 10 (smallest)

    df = df_data.loc[universe_, [date, 'PERMNO', 'CAPCO', 'EXCHCD']].set_index([date, 'PERMNO']).sort_index()
    df = df.groupby(level=0).apply(deciles).rename('SIZE_DECILE')
    while df.index.nlevels > 2:
        df = df.reset_index(level=0, drop=True)
    panel = Panel().set_frame(df, append=False).persist('SIZE_DECILE')
    print(panel.name, len(panel), panel.nlevels)
    

if __name__ == '__main__':

    # Load Fama-French factors
    #load_csv(DATA_LAKE / 'FF.csv', date_name='Date', sep='\t', 
    #         mul=0.01, append=False, keep=['Mkt-RF', 'HML'])

    # Load CRSP monthly data
    restrict_universe = True
    load_crsp(DATA_LAKE / 'crsp.txt.gz', date='date', sep='\t', restrict_universe=restrict_universe)

    # Load PSTAT annual data
    stock_name = 'gvkey'   # 'LPERMNO'
    date_name = 'datadate'
    sep = '\t'
    keep = ['txditc', 'seq', 'pstk', 'pstkrv', 'pstkl']
    filter = dict(indfmt = 'INDL', datafmt = 'STD', curcd = 'USD', popsrc = 'D', consol = 'C')
    aged = 2

    append = False  # to start loop over input files
    for subname in ['']:  #['2020', '2010']:
        filename = PSTAT_DATA / f"annual{subname}.txt.gz"
#        filename = DATA_LAKE / f"annual{subname}.txt.gz"
        load_panel_csv(filename, stock_name=stock_name, date_name=date_name, 
                       append=append, sep=sep, keep=keep, aged=aged, filter=filter)
        append = True

# For 
#for filename in tqdm(sorted((DATA_LAKE / 'USA').glob('19[8765432]*.txt.gz'), reverse=True)):
#    print(f"Loading {filename}")
#    load_panel_csv(filename, stock_name=STOCK_NAME, date_name=DATE_NAME, age=False, sep='\t')
