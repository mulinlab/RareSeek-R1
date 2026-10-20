"""Run standard-inference configurations and summarize evaluation metrics."""
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rareseek.io import load_config, path
from rareseek.pipeline import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    configs = [load_config(c) for c in a.configs]
    if len({c["output_dir"] for c in configs}) != len(configs):
        raise ValueError("Experiments must use different output directories")
    if any(not c.get("gold") or not c.get("graph") for c in configs):
        raise ValueError("Metric comparison requires gold and graph in every configuration")
    rows = []
    for config in configs:
        result = run(config)
        for metric in ("exact", "hierarchical"):
            for k, value in result[metric].items():
                rows.append({"configuration": Path(config["config_file"]).name, "backend": config["backend"],
                    "data_kind": config["data_kind"], "metric": metric, "k": k, "hits": value["hits"],
                    "total": result["total"], "accuracy": value["accuracy"], "missing": result["missing"], "empty": result["empty"]})
    output = path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
