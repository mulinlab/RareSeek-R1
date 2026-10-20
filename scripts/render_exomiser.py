"""Render the existing Exomiser configuration template without patient data."""
import argparse
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rareseek.io import ROOT, path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vcf", required=True)
    p.add_argument("--hpo", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if not path(a.vcf).is_file():
        p.error("VCF does not exist")
    if any(not re.fullmatch(r"HP:\d{7}", h) for h in a.hpo):
        p.error("HPO IDs must have form HP:0000001")
    template = (ROOT / "Exomiser_Config.yml").read_text(encoding="utf-8")
    output = path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("{vcf_path}", json.dumps(str(path(a.vcf))))
                      .replace("{hpo_ids}", json.dumps(a.hpo)), encoding="utf-8")
