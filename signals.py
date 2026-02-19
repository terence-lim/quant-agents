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

PSTAT_DATA = Path("/home/terence/Downloads/scratch/2024/PSTAT")
CRSP_DATA = Path("/home/terence/Downloads/scratch/2024/CRSP")
BENCH_DATA = DATA_LAKE / "factor-returns"


###########################
#
# load_panel_csv: for PSTAT and JKP characteristics
#
###########################


def load_panel_csv(
    filename: str | Path,
    stock_name: str,
    date_name: str,
    sep: str,
    append: bool,
    aged: int,
    filter: Dict = {},
    keep: List[str] = [],
) -> None:
    """Load each column of CSV file to its own Panel."""
    # if True:
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
    print("Dates not in calendar:", sorted(set(dates) - set(cal.dates.index)))

    # Apply filters
    df_data = df_data[df_data[date_name].isin(cal.dates.index)]
    for k, v in filter.items():
        if k in df_data.columns:
            df_data = df_data[df_data[k] == v]

    # Lookup stock identifiers by link date
    lookup = Lookup(source=stock_name)
    df_data[stock_name] = [
        lookup(s, d)
        for s, d in zip(df_data[stock_name].tolist(), df_data[date_name].tolist())
    ]
    df_data = df_data[df_data[stock_name] > 0]
    df_data.set_index([date_name, stock_name], inplace=True)
    df_data.index.names = (DATE_NAME, STOCK_NAME)

    for i, col in tqdm(enumerate(keep), total=len(keep)):
        if col not in df_data.columns:
            print(f"  - Skipping {col} as not in data")
            continue
        df = df_data[[col]].dropna().convert_dtypes()
        panel = Panel(col)
        print(panel.name, len(panel), panel.nlevels)
        panel = panel.set_frame(df, append=append)
        print("  - append:", panel.name, len(panel), panel.nlevels)
        if aged:
            age = panel.trend(cumcount)  # observation age
            panel = panel.filter(mask=(age >= aged)).persist(col)
            print(f"  - aged{aged}:", panel.name, len(panel), panel.nlevels)
        pprint(panel.info)
        # print(df.head())


if __name__ == "__main__":
    # Load Fama-French factors
    #    filename = 'F-F_Research_Data_Factors.csv'
    #    keep = ['Mkt-RF', 'HML', 'SMB', 'HML', 'RF']
    #    filename = 'F-F_Research_Data_5_Factors_2x3.csv'
    #    keep = ['RMW', 'CMA']
    #    load_csv(DATA_LAKE / filename, date_name='', sep=',',
    #             mul=0.01, append=False, keep=keep)

    # Compute stock excess returns EXCRET = RET - RF
    # ret = Panel('RET')
    # rf = Panel('RF')
    # Panel('EXCRET').set_frame((ret.frame.iloc[:,0] - rf.frame.iloc[:,0]).dropna()).persist('EXCRET')
    # # Load CRSP monthly data
    # restrict_universe = True
    # load_crsp(DATA_LAKE / 'crsp.txt.gz', date='date', sep='\t', restrict_universe=restrict_universe)

    # Load PSTAT annual data
    #     stock_name = 'gvkey'   # 'LPERMNO'
    #     date_name = 'datadate'
    #     sep = '\t'
    #     keep = ['txditc', 'seq', 'pstk', 'pstkrv', 'pstkl']
    #     keep = ['act', 'ebitda', 'ch', 'ebit', 'oiadp', 'caps', 'xsga']
    #     filter = dict(indfmt = 'INDL', datafmt = 'STD', curcd = 'USD', popsrc = 'D', consol = 'C')
    #     aged = 2

    #     append = False  # to start loop over input files
    #     for subname in ['']:  #['2020', '2010']:
    #         filename = PSTAT_DATA / f"annual{subname}.txt.gz"
    # #        filename = DATA_LAKE / f"annual{subname}.txt.gz"
    #         load_panel_csv(filename, stock_name=stock_name, date_name=date_name,
    #                        append=append, sep=sep, keep=keep, aged=aged, filter=filter)
    #         append = True

    # For
    # for filename in tqdm(sorted((DATA_LAKE / 'USA').glob('19[8765432]*.txt.gz'), reverse=True)):
    #    print(f"Loading {filename}")
    #    load_panel_csv(filename, stock_name=STOCK_NAME, date_name=DATE_NAME, age=False, sep='\t')
    pass
