# python evaluate_agent.py test_book_value book_value_       
# (c) Terence Lim
import argparse
import ast
import json
from pathlib import Path
from statistics import median

from qrafti import Panel
from utils import OUTPUT

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
    return panel_ids


def pearson_against_ground(panel_id: str, ground: Panel) -> float:
    try:
        panel = Panel().load(panel_id)    
        both = panel.frame.dropna().join(ground.frame.dropna(), how="inner", rsuffix="_ground")
        return round(float(both.corr(method="pearson").fillna(0).iloc[0, 1]), 4)
    except:
        print('Error in', panel_id)
        return 0.0


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
    pearsons = [pearson_against_ground(panel_id, ground) for panel_id in responses]

    avg = sum(pearsons) / len(pearsons)
    med = median(pearsons)
    pass_at_1 = pearsons[0]
    pass_at_3 = max(pearsons[:3])
    pass_at_k = max(pearsons)
    
    output_path = OUTPUT / "tests.csv"
    line = (f"{pass_at_1:.4f} | {pass_at_3:.4f} | {pass_at_k:.4f} | "
            f"{args.stem} | {avg:.4f} | {med:.4f} | {pearsons}\n")
    with output_path.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line)
