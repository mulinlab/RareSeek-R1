"""Normalize ranked names against a pinned local graph."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rareseek.io import path, load_json, read_jsonl, write_jsonl
from rareseek.pipeline import normalize_names
from rareseek.graph import validate_graph
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("input", "output", "graph"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    graph = validate_graph(load_json(path(a.graph)))
    rows = read_jsonl(path(a.input))
    for row in rows:
        row["predict_code"] = normalize_names(row["extracted_diseases"], graph)
    write_jsonl(path(a.output), rows)
