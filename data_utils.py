# data_utils.py  (c) Terence Lim 2025

from utils import as_nptype, DATA, BENCHMARKS_RAG, CHARACTERISTICS_RAG, RAG_PATH
from qrafti import Panel, STOCK_NAME, DATE_NAME
from rag import RAG

from pathlib import Path
import pandas as pd
from pandas.api.types import is_integer_dtype
import numpy as np
from typing import List, Dict, Any, Union
from tqdm import tqdm
import io
import re
from pprint import pprint, pformat

import logging

logging.basicConfig(level=logging.DEBUG, force=True)
#logging.disable(logging.CRITICAL)  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# DATA LAKE PATHS for raw data files
DATA_LAKE = Path("lake")
FF = DATA_LAKE / "FF"
CRSP = DATA_LAKE / "CRSP"
PSTAT = DATA_LAKE / "PSTAT"
JKP = DATA_LAKE / "JKP"

def load_rag(definitions: pd.Series, rag: str, build: bool = False):
    if build:
        RAG(rag, out_dir=RAG_PATH).build(definitions)
    else:
        RAG(rag, out_dir=RAG_PATH).load().add_documents(definitions, overwrite=True)
    return pd.read_parquet(RAG_PATH / rag / "docs.parquet")

def load_definitions(definitions_path: str, sep: str = "\t", keep: list = [], add: dict = {}) -> pd.Series:
    df = pd.read_csv(definitions_path, sep="\t", header=0)

    # infer name and description columns
    name_col, description_col = '', ''
    for col in df.columns:
        if "name" in col.lower():
            name_col = col
        if "description" in col.lower():
            description_col = col
    if not name_col:
        raise Exception(f"name column not in {df.columns.to_list()}")
    if not description_col:
        raise Exception(f"description column not in {df.columns.to_list()}")
    df = df.set_index(name_col)[description_col]

    # keep rows
    if keep:
        df = df[df.index.isin(keep)]
        
    # remove substring between parentheses
    df = df.apply(lambda x: re.sub(r"^\(.*?\)\s*", "", x))

    df = pd.concat([df, pd.Series(add).rename(description_col)])
    return df


#################
#
# load_fama_french - FF factor returns
#
#################

def load_fama_french(filename: str, sep: str, definitions: dict = {}, mul: float = 0.01, build: bool = False) -> None:
    """Load each column of CSV file to its own Panel."""
    # read all lines from filename, and check if first work of every line is a date
    with open(filename, "r") as f:
        lines = f.readlines()
    date_lines = []
    for line in lines:
        words = line.strip().split(sep)
        if len(words) == 0:
            continue
        first_word = words[0].strip()
        # is header row with labels
        if first_word == "" and len(words) > 2 and not date_lines:
            date_lines.append(line)
        if len(first_word) == 6:
            try:
                month_end = (pd.to_datetime(first_word, format="%Y%m") + pd.offsets.MonthEnd(0))\
                    .strftime("%Y-%m-%d")
                date_lines.append(line)
            except:
                pass

    # read lines as CSV into data frame
    df_data = pd.read_csv(
        io.StringIO("".join(date_lines)),
        sep=sep,
        header=0,
        index_col=0,
        low_memory=False,
    )

    df_data.index = pd.to_datetime(df_data.index.tolist(), format="%Y%m") + pd.offsets.MonthEnd(0)

    for i, col in tqdm(enumerate(definitions.keys()), total=len(definitions)):
        df = df_data[col].dropna().astype(float) * mul
        df.index.name = DATE_NAME
        panel = Panel(df).save(col)
        logging.info(f"{panel.name=}, {len(panel)=}, {panel.nlevels=}")

    load_rag(pd.Series(definitions), BENCHMARKS_RAG, build=build)
    return df_data


######################
#
# load_crsp - for CRSP Monthly
#
######################


crsp_definitions = dict(
    capco="Total market capitalization of the company as the sum of the capitalization of its stock classes, in millions",
    exchcd="Exchange Code, two-digit code indicating the exchange on which a security is listed (NYSE=1, AMEX=2 and Nasdaq=3)",
    total_count="Total count of all domestic stocks traded in the major US stock exchanges",
    total_cap="Total market capitalization of all domestic stocks traded in the major US stock exchanges",
    mthexcret="Monthly total stock return in excess of risk-free rate",
    # SIZE_DECILE="Size decile category of a stock, where break points are determined by NYSE stocks, descending size from 1 to 10 with 1 being the largest stocks",
)

"""
filename: str = CRSP / "monthly.txt.gz"
sep: str = "\t"
"""
def load_crsp(filename: str = CRSP / "monthly.txt.gz", sep: str = "\t", build: bool = True) -> pd.DataFrame:
    """Load each column of CSV file to its own Panel."""
    date_name: str = "mthcaldt"
    keep_float = ['mthcap', 'mthprc', 'mthret', 'mthretx', 'mthvol', 'shrout']
    keep_int = ['siccd']

    df_data = pd.read_csv(filename, sep=sep, header=0, low_memory=False)
    df_data.columns = [col.lower() for col in df_data.columns]
    print(len(df_data))
    df_data = df_data.loc[(df_data["sharetype"] == "NS") &
                          (df_data["securitytype"] == "EQTY") &
                          (df_data["securitysubtype"] == "COM") &
                          (df_data["usincflg"] =="Y") &
                          (df_data["issuertype"].isin(['ACOR', 'CORP'])) &
                          (df_data["primaryexch"].isin(['N','A', 'Q'])) &
                          (df_data["conditionaltype"] == 'RW') &
                          (df_data["tradingstatusflg"] == 'A'),
                          keep_float + keep_int + [date_name, 'permno', 'permco', 'primaryexch']]
    print(len(df_data))

    # Coerce numeric values and dropna()
    df_data[date_name] = pd.to_datetime(df_data[date_name], format="%Y-%m-%d", errors="coerce") + pd.offsets.MonthEnd(0)
    for col in keep_float:
        df_data[col] = pd.to_numeric(df_data[col], errors="coerce").astype(float)
    for col in keep_int:
        df_data[col] = pd.to_numeric(df_data[col], errors="coerce").fillna(0).astype(int)

    # CAPCO: sum up MthCap by date and PERMCO, then merge back to df_data
    cap = df_data.groupby([date_name, "permco"])["mthcap"].sum().rename("capco")
    df_data = df_data.merge(cap, on=[date_name, "permco"], how="left")
                           
    # EXCHCD
    df_data["exchcd"] = df_data["primaryexch"].map({"N": 1, "A": 2, "Q": 3}).fillna(0).astype(int)

    # Save to panels
    def save_frame(df: pd.DataFrame):
        """helper to save a panel"""
        df.set_index([date_name, "permno"], inplace=True)
        df.index.names = (DATE_NAME, STOCK_NAME)
        col = df.columns[0]
        if is_integer_dtype(df[col]):
            df = df[df[col] > 0].dropna()
        else:
            df = df.dropna()
        panel = Panel(df).save(col)
        logging.info(f"{panel.name=}, {len(panel)=}, {panel.nlevels=}")
        return panel
    for col in tqdm(keep_float + keep_int + ["capco", "exchcd"]):
        save_frame(df_data[[date_name, "permno", col]])

    cap_df = df_data[[date_name, "permno", "mthcap", "exchcd"]].dropna().set_index([date_name, "permno"]).sort_index()
        
    # Assign into size deciles based on stocks with exchcd == 1 (NYSE), needed in write_report()
    def deciles(x: pd.DataFrame) -> pd.DataFrame:
        """Assign x['MthCap'] into descending decile ranks using break points where x['EXCHCD'] is 1"""
        nyse_ = x["exchcd"] == 1  # NYSE
        breakpoints = x.loc[nyse_, "mthcap"].quantile(np.linspace(0, 1, 11)).to_numpy().copy()
        logging.debug(f"{len(x)=}, {x.iloc[0]}, {breakpoints=}")        
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        ranks = pd.cut(x["mthcap"], bins=breakpoints, labels=range(10, 0, -1), include_lowest=True)
        return ranks.astype(int)  # decile ranks from 1 (largest) to 10 (smallest)
    size_df = cap_df.groupby(level=0).apply(deciles).rename("size_decile")
    while size_df.index.nlevels > 2:
        size_df = size_df.reset_index(level=0, drop=True)
    panel = save_frame(size_df.reset_index())
    logging.info(f"{panel.name=}, {len(panel)=}, {panel.nlevels=}")

    # Compute and persist total market cap and count of stocks in universe
    def save_series(df: pd.Series, col: str):
        df.index.name = DATE_NAME
        df = pd.DataFrame(df.rename(col))
        panel = Panel(df).save(col)
        logging.info(f"{panel.name=}, {len(panel)=}, {panel.nlevels=}")        
        return panel
    
    cap = save_series(cap_df.groupby(level=0)["mthcap"].sum(), "total_cap")
    count = save_series(cap_df.groupby(level=0).apply(len), "total_count")

    # Compute stock excess returns EXCRET = RET - RF
    ret = Panel().load("mthret", start_date="1900-01-01", end_date="2100-12-31")
    rf = Panel().load("RF", start_date="1900-01-01", end_date="2100-12-31")
    excret = ret.restrict(subset=rf) - rf
    excret.save("mthexcret")

    definitions_path = RAG_PATH / "CRSP.csv"
    definitions = load_definitions(definitions_path, keep=keep_int + keep_float, add=crsp_definitions)
    load_rag(definitions, CHARACTERISTICS_RAG, build=build)
    # RAG(CHARACTERISTICS_RAG, out_dir=RAG_PATH).build(pd.Series(crsp_definitions))
    
    return df_data


###########################
#
# load JKP benchmarks and characteristics
#
###########################
def pivot_csv(df: pd.DataFrame, values: str, index: str, columns: str, keep: List = []) -> pd.DataFrame:
    """Helper to read from (benchmark returns) CSV file and pivot to columns"""
    df = df.pivot(index=index, columns=columns, values=values).sort_index()
    if keep:
        df = df[keep]
    df = df.convert_dtypes()
    df.index.name = DATE_NAME
    return df


def load_jkp(ret_type: str):
    definitions_path = RAG_PATH / "JKP.csv"
    benchmarks_path = JKP / f"[usa]_[all_factors]_[monthly]_[{ret_type}].csv"
    characteristics_folder: str = JKP

    # Read benchmarks
    df = pd.read_csv(benchmarks_path, sep=",", header=0, low_memory=False)
    benchmarks_df = pivot_csv(df, values="ret", index="date", columns="name")
    
    # coerce index date
    benchmarks_df.index = pd.to_datetime(benchmarks_df.index, format="%Y-%m-%d", errors="coerce")
    benchmarks_df.index.name = DATE_NAME
    keep = benchmarks_df.columns.tolist()
    
    # save benchmark definitions to RAG
    rag_df = load_definitions(definitions_path, keep=keep)
    rag_df.index = rag_df.index + "_" + ret_type
    rag_df = "JKP monthly benchmark returns for factor: " + rag_df
    load_rag(rag_df, BENCHMARKS_RAG)

    # Save benchmark returns as panels
    benchmarks_df.columns = benchmarks_df.columns + "_" + ret_type
    for bench in benchmarks_df.columns:
        df = benchmarks_df[[bench]].dropna().astype(float)
        Panel(df).save(bench)

    return benchmarks_df

###########################
#
# load_pstat: for PSTAT characteristics
#
###########################

class Lookup:
    def __init__(self, source: str = "gvkey", sep="\t"):
        """Initialize a Lookup object for mapping identifiers over time
        Args:
            filename (str | Path): Path to the lookup CSV file
            source (str): Column name for source identifiers
        Notes:
            # TODO: Handle dtypes of source and targe!
        """
        if source == "gvkey":
            filename = PSTAT / "links.txt.gz"
            target_name = "lpermno"
            date_name = "linkdt"
        if source == "permno":
            filename = CRSP / "names.txt.gz"
            target_name = "permco"
            date_name = "date"

        df = pd.read_csv(filename, sep=sep, header=0, low_memory=False).convert_dtypes()
        df.columns = [col.lower() for col in df.columns]
        for fmt in ["%Y-%m-%d", "%Y%m%d"]:
            try:
                df[date_name] = pd.to_datetime(df[date_name].astype(str), format=fmt, errors="coerce")
                logging.debug(f"Converted {date_name} using format {fmt}")
                break
            except Exception:
                logging.debug(f"Failed to convert {date_name} using format {fmt}")
        df = as_nptype(df.sort_values([source, date_name]).dropna(subset=[source, target_name, date_name]))
        try:
            df = df.loc[df[source] > 0]
        except Exception:
            # Fallback for non-numeric source identifiers (e.g., strings)
            df = df.loc[df[source].str.len() > 0]
        self.lookups = df.groupby(source)
        self.keys = set(self.lookups.indices.keys())
        self.source = source
        self.target = target_name
        self.date = date_name

    def __call__(self, stock: str, date: str = "2099-12-31", target: str = None) -> Any:
        """Return target identifiers matched to source as of date"""
        target = target or self.target
        if isinstance(date, str):
            date = pd.to_datetime(date, format="%Y-%m-%d", errors="coerce")
        if stock in self.keys:
            a = self.lookups.get_group(stock)
            b = a[a[self.date] <= date].sort_values(self.date)
            return b.iloc[-1].at[target] if len(b) else 0
        else:
            return 0


# For Fama-French Universe with observation age >= 2 years
def cumcount(x: pd.DataFrame) -> pd.Series:
    """
    Compute the cumulative count of observations
    Arguments:
        x: DataFrame with at least one column
    Returns:
        pd.Series with the cumulative count of observations
    Usage:
        panel.trend(cumcount)
    """
    return pd.Series(np.arange(len(x)), index=x.index)


source = "annual"
subnames = ["2020"]

def load_pstat(source: str, subnames: List[str],
               drop: List[str] = ["costat", "curcd", "datafmt", "indfmt", "consol", "tic", "fyr", "fyear",
                                  "curcdq", "fqtr", "rdq"]) -> None:
    """Load each column of CSV file to its own Panel."""
    stock_name = "gvkey"
    date_name = "datadate"
    sep = "\t"
    definitions_path = RAG_PATH / f"{source.upper()}.csv"
    keep_int = ["sich"]

    # 1. Read characteristics
    append = False  # to start loop over input files
    for subname in subnames:  # ['2020', '2010']:
        filename = PSTAT / f"{source.lower()}{subname}.txt.gz"
        df_data = pd.read_csv(filename, sep=sep, header=0, low_memory=False)

        # Drop unwanted columns, except stock_name and date_name
        drop = list(set(drop) - {stock_name, date_name})
        df_data = df_data.drop(columns=drop, errors="ignore")

        # Convert dates to datetime
        for fmt in ["%Y-%m-%d", "%Y%m%d"]:
            try:
                df_data[date_name] = pd.to_datetime(df_data[date_name], format=fmt) + pd.offsets.MonthEnd(0)
                logging.debug(f"Converted {date_name} using format {fmt}")
                break
            except Exception:
                logging.debug(f"Failed to convert {date_name} using format {fmt}")

        # Lookup stock identifiers by link date
        lookup = Lookup(source=stock_name)
        df_data[stock_name] = [lookup(s, d) for s, d in zip(df_data[stock_name].tolist(),
                                                            df_data[date_name].tolist())]
        df_data = df_data[df_data[stock_name] > 0]
        df_data.set_index([date_name, stock_name], inplace=True)
        df_data.index.names = (DATE_NAME, STOCK_NAME)

        keep = df_data.columns.to_list()
        for i, col in tqdm(enumerate(keep), total=len(keep), desc=f"Panels from {str(filename)}"):
            if col in keep_int:
                df = df_data[[col]].fillna(0).astype(int)
            else:
                df = as_nptype(df_data[[col]].dropna())
            panel = Panel(df)
            logging.debug(f"{panel.name=}, {len(panel)=}, {panel.nlevels=}")
            if append:
                old_panel = Panel().load(col, start_date="1900-01-01", end_date="2100-12-31")
                panel = old_panel.append(panel)
                logging.debug(f"  - append: {panel.name=}, {len(panel)=}, {panel.nlevels=}")
            panel.save(col)
        append = True

    # 2. Read definitions
    rag_df = load_definitions(definitions_path, keep=keep)
    rag_df = rag_df + f" (Source: Compustat {source.capitalize()})"
    load_rag(rag_df, CHARACTERISTICS_RAG)

    return df_data


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("error", category=RuntimeWarning)

    ###########################
    #
    # 1. Load Fama-French factors
    #
    ###########################

    def do_load_fama_french(build=True):
        filename = 'F-F_Research_Data_Factors.csv'
        definitions = {
            'Mkt-RF':"Monthly excess return on the market",
            'HML': "Fama-French HML factor constructed as the average return on two value portfolios minus the average return on two growth portfolios",
            'SMB': "Fama-French SMB factor constructed as the average return on three small portfolios minus the average return on three big portfolios",
            'RF': "Risk-free return as the one-month Treasury bill rate"
        }
        df = load_fama_french(FF / filename, sep=',', definitions=definitions, build=build)

        filename = 'F-F_Research_Data_5_Factors_2x3.csv'
        definitions = {
            'RMW': "Fama-French RMW factor constructed as the average return on two robust profitability portfolios minus the average return on two weak profitability portfolios",
            'CMA': "Fama-French CMA factor constructed as the average return on two conservative investment portfolios minus the average return on two aggressive investment portfolios"
        }
        df = load_fama_french(FF / filename, sep=',', definitions=definitions, build=False)


    ###########################
    #
    # Load CRSP monthly data
    #
    ###########################

    def do_load_crsp(build=True):
        df_data = load_crsp(build=build)
        return df_data
        #cap_df, cap_panel, count_df, count_panel, df_cap = df_data  ###
        #raise Exception
    
    ###########################
    #
    # Load PSTAT Annual and Quarterly characteristics
    #
    ###########################
    def do_load_pstat():
        pstat_df = load_pstat(source="Annual", subnames=["2020", "2010", "2000", "1990", "1950"])
        # pstat = load_pstat(source="Quarterly", subnames=["2010", "2000", "1990", "1960"])
        return pstat_df
    
    ###########################
    #
    # Load JKP benchmarks and characteristics
    #
    ###########################
    def do_load_jkp():
        return load_jkp(ret_type="vw_cap")

    #do_load_fama_french(build=True)
    #definitions_path = RAG_PATH / "JKP.csv"
    #df = load_definitions(definitions_path)

    #df = do_load_jkp()
    #rag_df = pd.read_parquet(RAG_PATH / BENCHMARKS_RAG / "docs.parquet")

    crsp_df = do_load_crsp(build=False)
    #pstat_df = do_load_pstat()
