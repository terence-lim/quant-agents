# search_logs.py
# TO DO: add optional list of input_args keys to include in graph nodes, e.g. "month", "cuts" 
import json
import os
from typing import Dict, Any, List, Set, Tuple
from collections import deque
import re
import html
from datetime import datetime
from graphviz import Source

from server_utils import TOOLS_LOGFILE, CODES_LOGFILE
from utils import DataCache, OUTPUT

SUBGRAPH_PNG = str(OUTPUT / "subgraph.png")

def store_conversation(debug_text: str = ''):
    with (open(OUTPUT / "conversation.txt", 'w') as f):
        f.write(f"=== {str(datetime.now())} ===\n")
        f.write(debug_text)
        f.flush()

def restart(logname: str = TOOLS_LOGFILE):
    DataCache.reset()
    # write to "tools.log"
    with open(logname, "w") as f:
        f.write("")


def load_recent_code_logs(
        max_items: int = 5,
        filename: str = CODES_LOGFILE
) -> List[dict]:
    """Load up to max_items most recent JSON code-log objects."""
    if not filename or not os.path.exists(filename):
        return []

    records: List[dict] = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            try:
                obj = json.loads(row)
            except json.JSONDecodeError:
                # Skip legacy/non-JSON entries.
                continue
            if isinstance(obj, dict) and "date" in obj and "code_str" in obj:
                records.append(obj)

    records = sorted(records, key=lambda x: x.get("date", ""), reverse=True)
    return records[:max_items]


def load_objects(filename: str = TOOLS_LOGFILE) -> Dict[str, dict]:
    """
    Read a file containing multiple JSON objects (concatenated, possibly spanning
    multiple lines), keep only those that have an "output" dict with a
    "results_panel_id" key, and return a dict keyed by that results_panel_id.

    The file format is assumed to be:
        { ... }{ ... }{ ... }
    or with arbitrary whitespace/newlines between/inside objects, e.g.:
        {
          "output": {
            "results_panel_id": "A",
            ...
          }
        }
        {
          "output": {
            "results_panel_id": "B",
            ...
          }
        }

    This function does *not* require one JSON object per line.

    Parameters
    ----------
    filename : str
        Path to the input file containing concatenated JSON objects.

    Returns
    -------
    objects_by_id : dict[str, Any]
        A dictionary mapping each results_panel_id (string) to its corresponding
        parsed JSON object (a Python dict).
    """
    objects_by_id: Dict[str, Any] = {}

    with open(filename, "r", encoding="utf-8") as f:
        buffer_chars: list[str] = []
        depth = 0
        in_string = False
        escape = False

        def try_process_buffer():
            """Parse the current buffer as JSON and, if valid, filter & store."""
            nonlocal buffer_chars, objects_by_id
            raw = "".join(buffer_chars).strip()
            buffer_chars = []  # reset buffer

            if not raw:
                return

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # Invalid JSON blob; skip silently or log if desired
                return

            # Filter: keep only if obj["output"]["results_panel_id"] exists
            if isinstance(obj, dict):
                output = obj.get("output")
                if isinstance(output, dict):
                    results_panel_id = output.get("results_panel_id")
                    if isinstance(results_panel_id, str) and results_panel_id:
                        objects_by_id[results_panel_id] = obj

        for line in f:
            for ch in line:
                # Always accumulate the character *after* we’ve decided if we’re
                # starting a new object, etc.
                if depth == 0:
                    # Ignore characters until we see a top-level '{'
                    if ch.isspace():
                        continue
                    if ch != "{":
                        # Non-whitespace and not '{' outside an object → skip
                        continue
                    # Starting a new object
                    buffer_chars = ["{"]
                    depth = 1
                    in_string = False
                    escape = False
                    continue

                # We are inside a JSON object: accumulate
                buffer_chars.append(ch)

                # Handle string/escape state machine for correct brace tracking
                if escape:
                    # Current char is escaped; just consume it
                    escape = False
                    continue

                if ch == "\\":
                    # Next char will be escaped if we're in a string
                    if in_string:
                        escape = True
                    continue

                if ch == '"':
                    # Toggle string state
                    in_string = not in_string
                    continue

                # Only track braces when NOT inside a string
                if not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        # If depth hits 0, we've closed the top-level object
                        if depth == 0:
                            try_process_buffer()

        # In case file ends while still holding a complete object at depth 0
        if depth == 0 and buffer_chars:
            try_process_buffer()

    return objects_by_id


def traverse_links(objects: Dict[str, dict], start_id: List[str] = None) -> List[str]:
    """
    Breadth-first traversal over linked panel objects starting from multiple roots.

    Enhancements:
    1) Accept a list of start_ids. Traversal is global: nodes visited from earlier
       start_ids are not re-visited.
    2) If no start_ids are given, compute all root nodes automatically. A root
       node is defined as a node that never appears as a panel_id in ANY input
       field containing the substring 'panel_id'.
    3) Visit each node at most once. Uses BFS order.
    """

    # ------------------------------------------------------------------
    # Helper: extract neighbors (panel links)
    # ------------------------------------------------------------------
    def find_neighbors(obj: dict) -> List[str]:
        """Return all linked panel IDs (strings) via any '*panel_id*' key."""
        nbrs = []
        inp = obj.get("input", {})
        for key, val in inp.items():
            if "panel_id" in key and val != "":
                if isinstance(val, str):
                    nbrs.append(val)
                elif isinstance(val, list):
                    nbrs.extend([v for v in val if isinstance(v, str)])
        return nbrs

    # ------------------------------------------------------------------
    # If start_id is not provided, discover all root nodes
    # ------------------------------------------------------------------
    if not start_id:
        # Collect all nodes that appear as panel targets
        pointed_to: Set[str] = set()
        for obj in objects.values():
            for pid in find_neighbors(obj):
                pointed_to.add(pid)

        # Roots = all nodes not pointed to
        start_id = [nid for nid in objects.keys() if nid not in pointed_to]

    # Convert to list if single value was passed
    if isinstance(start_id, str):
        start_id = [start_id]

    # ------------------------------------------------------------------
    # BFS traversal from multiple starting nodes
    # ------------------------------------------------------------------
    visited: Set[str] = set()
    order: List[str] = []

    queue: deque[Tuple[str, int]] = deque()

    # Seed queue with all starting nodes at depth 0
    for sid in start_id:
        queue.append((sid, 0))

    # ------------------------------------------------------------------
    # BFS main loop
    # ------------------------------------------------------------------
    while queue:
        current, depth = queue.popleft()

        if current in visited or current not in objects:
            continue

        visited.add(current)
        obj = objects[current]
        indent = "    " * depth
        node = f"{indent}{current:5s}"

        input_items = obj.get("input", {}).items()
        output_items = obj.get("output", {}).items()

        input_args = {k: v for k, v in input_items if "panel_id" not in k}
        node += f" = {obj.get('tool', '')}{str(input_args)}"

        output_args = {k: v for k, v in output_items if k in ["nlevels", "rows"]}
        if output_args:
            node += f" -> {str(output_args)}"

        panel_args = [
            i
            for j in [
                v if isinstance(v, list)
                else [(v if isinstance(v, str) else f"{(v)}")]
                for k, v in input_items
                if "panel_id" in k and v is not None and v != ""
            ]
            for i in j
        ]

        for panel_arg in panel_args:
            node += f"\n{indent}{' ' * 8}...{panel_arg}"

        order.append(node + "\n")

        # --------------------- add neighbors (BFS) ------------------------
        for nxt in panel_args:
            if isinstance(nxt, str) and nxt not in visited:
                queue.append((nxt, depth + 1))

    return order



def generate_graphviz(objects: Dict[str, dict], start_node: str = None) -> str:
    """
    Generate a Graphviz DOT representation of the panel dependency graph.

    Nodes use HTML-like <table> labels showing:
        - panel_id (title row)
        - tool name
        - all input arguments EXCEPT those containing 'panel_id'

    Edges:
        - have no label for "panel_id"
        - have a label for any other *panel_id field*, e.g. "other_panel_id"
    """

    # ----------------------------------------------------------------------
    # Helper: extract neighbors with field names
    # ----------------------------------------------------------------------
    def find_neighbors_with_fields(obj: dict) -> List[Tuple[str, str]]:
        """Return list of (target_panel_id, fieldname)."""
        neighbors = []
        inp = obj.get("input", {})

        for key, val in inp.items():
            if "panel_id" in key and val is not None and val != "":
                if isinstance(val, str):
                    neighbors.append((val, key))
                elif isinstance(val, list):
                    neighbors.extend((v, key) for v in val if isinstance(v, str))

        return neighbors

    # ----------------------------------------------------------------------
    # Determine set of nodes to include
    # ----------------------------------------------------------------------
    if not start_node:
        nodes_to_include = set(objects.keys())
    else:
        visited: Set[str] = set()
        stack = [start_node] if isinstance(start_node, str) else start_node

        while stack:
            curr = stack.pop()
            if curr in visited or curr not in objects:
                continue
            visited.add(curr)

            for nxt, _field in find_neighbors_with_fields(objects[curr]):
                if nxt not in visited:
                    stack.append(nxt)

        nodes_to_include = visited

    # ----------------------------------------------------------------------
    # Begin DOT
    # ----------------------------------------------------------------------
    lines = ['digraph PanelGraph {']
    lines.append('    rankdir=TB;')  # LR for left-to-right
    lines.append('    node [shape=plaintext, fontsize=10];')  # HTML table nodes

    # ----------------------------------------------------------------------
    # Emit nodes as HTML <table>
    # ----------------------------------------------------------------------
    for node in sorted(nodes_to_include):
        obj = objects.get(node, {})
        input_args = obj.get("input", {})
        tool = obj.get("tool", "")
        output_args = obj.get("output", {})
        output_suffix = ""
        if 'nlevels' in output_args:
            output_suffix += f" [{output_args['nlevels']}]"
        if 'rows' in output_args:
            output_suffix += f" ({output_args['rows']:,})"

        # Sanitize for DOT object name
        safe_node = re.sub(r"[^a-zA-Z0-9_]", "_", node)

        # Escape HTML chars in values
        safe_tool = html.escape(str(tool).replace('Panel_', ''))

        # Build table rows
        rows = []

        # Title row = panel ID
        rows.append(
            f'<tr><td colspan="2" bgcolor="#D0D0FF"><b>{html.escape(node)}</b>{output_suffix}</td></tr>'
        )

        # Tool row
        rows.append(
            f'<tr><td align="left">tool</td><td align="left"><b>{safe_tool}</b></td></tr>'
        )

        # ---------------------------------------------------------
        # Detect *missing* panel links from ANY input key containing "panel_id"
        # ---------------------------------------------------------
        missing_links = []

        for key, value in input_args.items():
            if "panel_id" not in key or value is None or value == "":
                continue

            # value may be a string or list of strings
            panel_ids = []
            if isinstance(value, list):
                panel_ids.extend([v for v in value if isinstance(v, str)])
            else:
                panel_ids.append(str(value))

            # Check each target against known nodes
            for pid in panel_ids:
                if pid not in nodes_to_include:
                    missing_links.append((key, pid))

        # Add missing-link rows (one per missing reference)
        for key, pid in missing_links:
            if key == "panel_id":
                safe_key = html.escape(key)
                color = "red"
            else:
                safe_key = html.escape(key.replace("_panel_id", ""))
                color = "blue"
            rows.append(
                f'<tr><td align="left"><font color="{color}">{safe_key}</font></td>'
                f'<td align="left"><font color="{color}">{html.escape(pid)}</font></td></tr>'
            )

        # --------------------------------------------
        # Add all non-panel_id input args normally
        # --------------------------------------------
        for key, val in input_args.items():
            if "panel_id" in key:
                continue  # skip here, handled above
            rows.append(
                f'<tr><td align="left"><font color="darkgreen">{html.escape(key)}</font></td>'
                f'<td align="left"><font color="darkgreen">{html.escape(str(val))}</font></td></tr>'
            )

        # Make the HTML table
        table = (
            f'<<table border="1" cellborder="0" cellspacing="0" cellpadding="4">'
            f'{"".join(rows)}'
            f'</table>>'
        )
        lines.append(f'    "{safe_node}" [label={table}];')

    # ----------------------------------------------------------------------
    # Emit edges with selective labels
    # ----------------------------------------------------------------------
    for node in sorted(nodes_to_include):
        if node not in objects:
            continue

        obj = objects[node]
        safe_src = re.sub(r"[^a-zA-Z0-9_]", "_", node)

        for neighbor, field_name in find_neighbors_with_fields(obj):
            if neighbor not in nodes_to_include:
                continue

            safe_dst = re.sub(r"[^a-zA-Z0-9_]", "_", neighbor)

            # Standard panel_id → red arrow
            if field_name == "panel_id":
                lines.append(
                    f'    "{safe_src}" -> "{safe_dst}" '
                    f'[dir=back, color="red", fontcolor="red"];'
                )
            else:
                # Nonstandard panel_id field → blue arrow & blue label
                safe_field = html.escape(field_name.replace("_panel_id", ""))
                lines.append(
                    f'    "{safe_src}" -> "{safe_dst}" '
                    f'[dir=back, color="blue", fontcolor="blue", label="{safe_field}"];'
                )
            # # Label only if field name != "panel_id"
            # if field_name == "panel_id":
            #     lines.append(f'    "{safe_src}" -> "{safe_dst}" [dir=back];')
            # else:
            #     safe_field = html.escape(field_name)
            #     lines.append(
            #         f'    "{safe_src}" -> "{safe_dst}" [label="{safe_field}" dir=back];'
            #     )

    lines.append("}")
    return "\n".join(lines)

# ---------------- EXAMPLE USAGE ----------------

def generate_dot(objects, start_key):
    path = traverse_links(objects, start_key)
    # print("".join(path[::-1]))  # Print in visit order

    #dot = generate_graphviz(objects)
    dot = generate_graphviz(objects, start_node=start_key)
    with open(OUTPUT / "subgraph.dot", "w") as f:
        f.write(dot)
    src = Source(dot)
    src.format = "png"
    src.render(str(OUTPUT / "subgraph"), cleanup=True)
    return SUBGRAPH_PNG

if __name__ == "__main__":
    objects = load_objects()

    print(f"Loaded {len(objects)} valid objects with results_panel_id keys.")

    # Example traversal from a known starting ID
    start_key = input("Enter ending results_panel_id (including leading undescore) for traversal: ").strip()
    print(generate_dot(objects, start_key))
