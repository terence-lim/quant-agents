import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
import markdown
from weasyprint import HTML
from pathlib import Path
from typing import List, Dict

#import logging
#logging.getLogger("fontTools").setLevel(logging.WARNING)
#logging.getLogger("matplotlib").setLevel(logging.WARNING)
#logging.getLogger("weasyprint").setLevel(logging.WARNING)
#logging.disable(logging.WARNING)  # disable all logging messages at or below

OUTPUT = Path("output")
LINKS = Path("LINKS")

BENCHMARKS_RAG = "benchmark_returns"
CHARACTERISTICS_RAG = "stock_characteristics"
#JKP_RAG_PATH = Path("/home/terence/Downloads/scratch/2024/JKP/JKP_RAG")
JKP_RAG_PATH = LINKS / "JKP_RAG"

#CRSP_RAG_PATH = Path("/home/terence/Downloads/scratch/2024/JKP/CRSP_RAG")
CRSP_RAG_PATH = LINKS / "CRSP_RAG"

# RAG_PATH = JKP_RAG # should move from qrafti.py

#MEDIA = Path("/home/terence/Downloads/scratch/2024/JKP/media")
MEDIA = LINKS / "media"

#WORKSPACE = Path("/home/terence/Downloads/scratch/2024/JKP/workspace")
WORKSPACE = LINKS / "workspace"

#
# Helper utilities
#
def as_nptype(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.copy()
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            try:
                df.loc[:, col] = df[col].astype(int)
            except Exception:
                df.loc[:, col] = df[col].astype(float)
        elif pd.api.types.is_float_dtype(df[col]):
            df.loc[:, col] = df[col].astype(float)
    return df

class suppress_stderr():
    def __enter__(self):
        self.original_stderr = sys.stderr
        self.devnull = open(os.devnull, 'w')
        sys.stderr = self.devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.original_stderr
        self.devnull.close()

def plt_savefig(savefig: str = '') -> str:
    """Helper to save a matplotlib figure with a timestamped filename in the MEDIA directory."""
    if not savefig:
        savefig = Path(MEDIA) / f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.png'
    plt.savefig(savefig)
    return 'file:///' + str(savefig)
        

def markdown_to_pdf(
    markdown_text: str,
    stylesheets: List[str] = ["style.css"],
    output_file: str = "output.pdf",
    debug: bool = False,
) -> Dict[str, str]:
    """Convert markdown text to PDF using WeasyPrint.
    Arguments:
        markdown_text: Markdown formatted string
        stylesheets: List of CSS stylesheet files to apply
        output_file: Output PDF file name
        debug: If True, print debug information
    """
    html_content = markdown.markdown(markdown_text, extensions=["tables"])
    if debug:
        print(html_content)
    html_doc = HTML(string=html_content)
    html_doc.write_pdf(output_file, stylesheets=stylesheets)
    return dict(output_file=output_file)

###########################
#
# Calendar of valid dates
#
###########################
class Calendar:
    def __init__(
        self,
        start_date: str = None, #DATES["start_date"],
        end_date: str = None, #DATES["end_date"],
        reference_panel: str = "TOTAL_COUNT"  # 'Mkt-RF' 'ret_exc_lead1m'
    ):
        # Initialize the Calendar with unique sorted dates from a reference Panel 'ret_exc_lead1m'
        dates = DataCache.read_frame(reference_panel).index.get_level_values(0)
        if start_date:
            dates = dates[dates >= start_date]
        if end_date:
            dates = dates[dates <= end_date]

        sorted_dates = sorted(dates.unique().tolist())
        self.start_date = sorted_dates[0]
        self.end_date = sorted_dates[-1]
        self.dates = pd.Series(range(len(sorted_dates)), index=sorted_dates)

    def dates_shifted(self, shift: int = 1) -> Dict[str, str]:
        """Return a mapping of original dates to shifted dates."""
        shifted_dates = dict()
        for i in range(len(self.dates)):
            date = self.dates.index[i] + pd.offsets.MonthEnd(shift)
            if date in self.dates:
                shifted_dates[self.dates.index[i]] = date
        return shifted_dates

    def dates_range(self, start_date: str, end_date: str) -> List[str]:
        """Return a list of dates in the calendar between start_date and end_date (inclusive)."""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        return [date for date in self.dates.index if start_date <= date <= end_date]

    def ismonth(self, date: str, months: List | int) -> bool:
        """Check if a date is in the specified months."""
        if isinstance(months, int):
            months = [months]
        month = int(str(date).split('-')[1])
        return month in months

    def offset(self, date: str, offset: int) -> str:
        """Return the date offset by the specified number of periods."""
        return pd.to_datetime(date) + pd.offsets.MonthEnd(offset)


        
###########################
#
# Data Cache library for persisting intermediate DataFrames
#
###########################
class DataCache:
    @staticmethod
    def load_cache() -> Dict[str, str]:
        """Load the data cache from the cache file."""
        cache_file = Path(WORKSPACE / "cache.json")
        try:
            with open(cache_file, "rb") as f:
                cache = json.load(f)
        except Exception:
            cache = {"file_id": 0}
        return cache

    @staticmethod
    def dump_cache(cache: Dict[str, str]):
        """Dump the data cache to the cache file."""
        cache_file = Path(WORKSPACE / "cache.json")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4)

    @staticmethod
    def write_frame(frame: pd.DataFrame, name: str = "") -> str:
        """Write a dataframe to parquet file in the data cache
        Arguments:
            frame: DataFrame to write
            name: Optional name for the file, if not given, a new name will be generated
        Returns:
            name: Name of the file (without extension)
        """
        if not name:  # temporary file name
            cache = DataCache.load_cache()
            file_id = int(cache.get("file_id", 0)) + 1
            name = f"_{file_id}"
            cache["file_id"] = file_id
            DataCache.dump_cache(cache)
        if frame is not None:
            frame.to_parquet(WORKSPACE / f"{name}.parquet", index=True)
        return name

    @staticmethod
    def read_frame(name: str) -> pd.DataFrame:
        """Read a dataframe from parquet file in the data cache
        Arguments:
            name: Name of the file (without extension)
        Returns:
            frame: DataFrame read from the file, None if not found
        """
        filename = WORKSPACE / f"{name}.parquet"
        if filename.exists():
            return as_nptype(pd.read_parquet(filename))
        else:
            return None

    @staticmethod
    def reset():
        """Clears and resets data cache"""
        for file in WORKSPACE.glob("_*.parquet"):
            file.unlink()
        cache_file = Path(WORKSPACE / "cache.json")
        if cache_file.exists():
            cache_file.unlink()


