# server_utils.py    (c) Terence Lim
import json
from datetime import datetime
import sys
import os
import subprocess
import time
import tempfile
from typing import Union, Dict

from qrafti import Panel
from rag import RAG
from utils import OUTPUT

from dotenv import load_dotenv
load_dotenv()
    
TOOLS_LOGFILE = str(OUTPUT / "tools.log")
CODES_LOGFILE = str(OUTPUT / "codes.log")

def now():
    return str(datetime.now())[:19]


###########################
#
# Log tool call messages
#
###########################
def log_tool(tool: str, input: dict = {}, output: dict = {}, mode: str = "a"):
    """Log tool call message to a log file."""
    message = json.dumps(dict(output=output, tool=tool, input=input, date=str(datetime.now())[:19]), 
                         indent=2)
    with open(TOOLS_LOGFILE, mode) as f:
        f.write(message + "\n"*2)
        f.flush()
    print(message)

def log_code(code_str: str, mode: str = "a"):
    """Server utility to keep log of code str executed"""
    message = json.dumps(dict(date=str(datetime.now())[:19], code_str=code_str))
    with open(CODES_LOGFILE, mode) as f:
        f.write(message + "\n")
        f.flush()


def invalid_panel(panel: Panel) -> str:
    """Return error message for invalid Panel."""
    if isinstance(panel, Panel) and panel.nlevels < 0:
        return f"Error: Panel {panel.name} does not exist."

    
###########################
#
# Memory/RAG
#
###########################
def query_rag(query: str, rag: RAG, top_n: int = 5) -> Dict[str, str]:
    """Retrieve top documents with query str from RAG"""
    query = query.strip()
    if query in rag.doc_series.index:
        return {query: rag.doc_series.loc[query]}
    results = rag.retrieve(query, top_n=top_n)
    descriptions = dict()
    for doc_id, text in zip(results['doc_id'], results["text"]):
        descriptions[doc_id] = text
    return descriptions


###########################    
#
# Safely converting input strings to types
#
###########################
def panel_or_numeric(x: str, **kwargs) -> Union["Panel", float, int]:
    """Convert a string to a Panel or numeric value or None"""
    if x is None or str(x).lower() in ["", "none"]:
        return None
    try:
        x = str(x)
        if "." in x:
            return float(x)
        else:
            return int(x)
    except Exception:
        start_date = None if x.startswith('_') else kwargs.get('start_date', None)
        end_date = None if x.startswith('_') else kwargs.get('end_date', None)
        panel = Panel().load(x, start_date=start_date, end_date=end_date)
        return panel

# All date and vulnerable str arguments for Panel methods should be converted using str_or_None
def str_or_None(x: str) -> Union[str, None]:
    """Convert a string to None if it is 'None' or empty."""
    if x is None or str(x).lower() in ["", "none"]:
        return None
    return str(x)


def numeric_or_None(x: str) -> Union[float, None]:
    """Convert a string to float else None"""
    try:
        return float(x)
    except Exception:
        return None

def bool_or_None(x: str) -> Union[bool, None]:
    """convert a string to bool else None"""
    if isinstance(x, str):
        x = x.lower()
        try:
            return bool(round(float(x)))
        except Exception:
            if x == "false":
                return False
            elif x == "true":
                return True
            else:
                return None
    else:
        if isinstance(x, (int, float, bool)):
            return bool(round(x))
        else:
            return None

    
def int_or_None(x: str) -> Union[float, None]:
    """Convert a string to int else None"""
    try:
        return int(float(x))
    except Exception:
        return None


###########################
#
# Running Python code
#
###########################
#def run_code_in_subprocess(code_str):
#    #with open("coding.log", "w") as f:
#    #    f.write(f"RUN CODE IN SUBPROCESS: {code_str}\n")
#    env = os.environ.copy()
    
#    # prepend your project root to PYTHONPATH
#    project_root = os.getenv("PROJECT_ROOT", "")
#    env["PYTHONPATH"] = (project_root + ":" + env.get("PYTHONPATH", "")).strip(":")
#    proc = subprocess.run(
#        [sys.executable, "-c", code_str], capture_output=True, text=True, env=env
#    )
#    print(f"Subprocess exited with code {proc.returncode}")
#    return proc.stdout, proc.stderr, proc.returncode

def run_code_in_subprocess(code_str: str, timeout: int = 900):
    env = os.environ.copy()

    project_root = os.getenv("PROJECT_ROOT")
    if project_root:
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            project_root
            if not old_pythonpath
            else project_root + os.pathsep + old_pythonpath
        )
#    print(env["PYTHONPATH"])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code_str)
        path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            env=env,
            #timeout=timeout,
        )
        print(f"Subprocess exited with code {proc.returncode}")
        print('stdout:', proc.stdout)
        print('stderr:', proc.stderr)
        return proc.stdout, proc.stderr, proc.returncode
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

if __name__ == "__main__":
    tic = time.time()

    # Test 1
    test_str = """
panel_id = 'me' 
from qrafti import Panel
import pandas as pd
p1 = Panel().load(f'{panel_id}') 
p2 = Panel().load('crsp_exchcd').apply(pd.DataFrame.isin, values=[1, '1'])
def winsorize(x, lower=0.0, upper=1.0) -> pd.Series: 
    lower, upper = x.loc[x.iloc[:, 1].astype(bool)].iloc[:, 0].quantile([lower, upper]).values
    return x.iloc[:, 0].clip(lower=lower, upper=upper)
p3 = p1.apply(winsorize, 1 if p2 is None else p2, lower=0, upper=0.8).persist()
print(p3.to_json())  # returns saved name of the panel as JSON string
""".strip()
    stdout, stderr, exit_code = run_code_in_subprocess(test_str)
    print(exit_code)
    print(stderr)
    print(stdout)


    # Test 2
    panel_id = "me"
    other_panel_id = "ret_exc_lead1m"
    code = f"""
import json
from qrafti import Panel
p1, p2 = Panel().load('{panel_id}'), Panel().load('{other_panel_id}')
p3 = (p1 @ p2).persist()
print(p3.to_json())  # returns saved name of the panel as JSON string
""".strip()
    stdout, stderr, exit_code = run_code_in_subprocess(code)
    print(exit_code)
    print(stderr)
    print(stdout)

    # Test 3
    code = """
# Please run this following code:
from qrafti import Panel, MEDIA
import matplotlib.pyplot as plt
ret = 'vw_cap'
factor = 'ret_12_1'
bench = Panel().load(factor + '_' + ret).frame
bench.cumsum().plot()
savefig = MEDIA / f"{factor}_{ret}.png"
plt.savefig(savefig)
print(f"[Image File](file:///{str(savefig)})")
""".strip()
    stdout, stderr, exit_code = run_code_in_subprocess(code)
    print(exit_code)
    print(stderr)
    print(stdout)


    toc = time.time()
    print(f"Total elapsed time: {toc - tic:.2f} seconds")

    
