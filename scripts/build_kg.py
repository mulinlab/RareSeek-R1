"""Run KG extraction stages with one consistent path convention."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rareseek.io import ROOT, path, write_json, digest
from rareseek.neo4j_csv import validate_csvs

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=os.getenv("RARESEEK_RAW_DIR", "RareMed-KG/data/raw"))
    p.add_argument("--processed", default=os.getenv("RARESEEK_PROCESSED_DIR", "RareMed-KG/data/processed"))
    a = p.parse_args()
    raw, processed = path(a.raw), path(a.processed)
    target = processed / 'neo4j_import'
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'build_manifest.json', {'status': 'running'})
    env = dict(os.environ, RARESEEK_RAW_DIR=str(raw), RARESEEK_PROCESSED_DIR=str(processed))
    for name in ("extract_orphanet.py", "extract_hpo_omim.py", "extract_variant.py", "load_neo4j.py"):
        subprocess.run([sys.executable, str(ROOT / "RareMed-KG/KG_construction" / name)], env=env, check=True)
    nodes, edges = validate_csvs(target, check_manifest=False)
    write_json(target / 'build_manifest.json', {
        'status': 'complete', 'raw': str(raw), 'processed': str(processed),
        'csv_sha256': {f.name: digest(f) for f in sorted(target.glob('*.csv'))},
        'source_sha256': {f.relative_to(raw).as_posix(): digest(f) for f in sorted(raw.rglob('*')) if f.is_file()},
        'variants_sha256': digest(processed / 'integration/variants_HGMD_Clinvar.tsv')})
    print("Validated graph:", {k: len(v) for k,v in nodes.items()})
