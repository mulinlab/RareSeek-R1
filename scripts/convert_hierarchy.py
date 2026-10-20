"""Convert the original OrphaCode/Children hierarchy to the evaluator contract."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rareseek.io import load_json, path, write_json

def convert(tree):
    parents = {}
    def visit(items, parent=None):
        if not isinstance(items, list):
            raise ValueError("Hierarchy must contain lists of node objects")
        for item in items:
            raw = item.get("OrphaCode")
            if raw is None:
                raise ValueError("Missing OrphaCode")
            code = str(raw)
            code = code if code.startswith("ORPHA:") else "ORPHA:" + code
            parents.setdefault(code, set())
            if parent:
                parents[code].add(parent)
            visit(item.get("Children", []), code)
    visit(tree)
    return {k: sorted(v) for k,v in sorted(parents.items())}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="Classifications of rare diseases.json")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    write_json(path(a.output), convert(load_json(path(a.input))))
