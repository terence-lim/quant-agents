# python evaluate_agent.py test_book_value book_value_       
# (c) Terence Lim
import argparse
import ast
import json
from pathlib import Path
from statistics import median
from datetime import datetime
import itertools
import numpy as np

from qrafti import Panel
from utils import OUTPUT

K = 5

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate response panels against a ground panel.")
    parser.add_argument("stem", help='Input filename stem in tests/, e.g. "test_book_value".')
    return parser.parse_args()


def _extract_response(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        payload = ast.literal_eval(line)

    response = payload.get("response", "")
    return str(response).strip()


def load_response_panel_ids(path: Path) -> list[str]:
    panel_ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            response = _extract_response(raw_line)
            if response:
                panel_ids.append(response)
    return panel_ids[-K:]   # return last K responses

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def similarity_against_ground(panel_id: str, ground: Panel, method="cosine") -> float:
    """cosine, pearson or spearman methods"""
    try:
        panel = Panel().load(panel_id)
        panel_df = panel.frame.dropna()
    except:
        print('Error in', panel_id)
        return 0.0
    if method == "cosine":
        method = cosine_similarity
    ground_df = ground.frame.dropna()
    how = "outer" if panel.nlevels == 2 else "left"
    both = panel_df.join(ground_df, how=how, rsuffix="_ground").fillna(0)
    print(f"{len(ground_df)=}, {len(panel_df)=}, {len(both)=}")
    if len(both):
        return round(float(both.corr(method=method).fillna(0).iloc[0, 1]), 4)
    else:
        return 0.0


def corr_k(correlations: list, k: int) -> float:
    """
    Compute corr@k_i from a list of precomputed correlations.

    Definition:
    For every size-k subsample of the N attempts, take the maximum
    correlation in that subsample. Then average these maxima across
    all subsamples.
    """
    correlations = list(correlations)
    N = len(correlations)
    max_corrs = []
    for subset in itertools.combinations(range(N), k):
        max_corrs.append(max(correlations[j] for j in subset))
    return float(np.mean(max_corrs))

if __name__ == "__main__":
    args = parse_args()
    input_path = OUTPUT / f"{args.stem}.responses"
    # ground_panel by removing "test_" prefix and appending '_' suffix
    ground_panel_id = args.stem[args.stem.find('_')+1:] + '_'  

    responses = load_response_panel_ids(input_path)
    if not responses:
        raise ValueError(f"No response panel IDs found in {input_path}")
    print(ground_panel_id, responses)

    ground = Panel().load(ground_panel_id)
    sims_list = [similarity_against_ground(panel_id, ground) for panel_id in responses]

    avg = sum(sims_list) / len(sims_list)
    med = median(sims_list)
    passes = [1, 2, 3, 5, len(sims_list)]
    vals = [corr_k(sims_list, k) for k in passes]
    
    output_path = OUTPUT / "tests.csv"
    date = str(datetime.now())[:19]
    line = (" & ".join([f"{v:.4f}" for v in vals]) +
            f"& {args.stem} & {avg:.4f} & {med:.4f} & {date} & {sims_list}\n")
    with output_path.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line)
