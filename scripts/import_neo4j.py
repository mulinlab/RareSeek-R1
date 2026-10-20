import argparse
import os
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rareseek.io import path
from rareseek.neo4j_csv import import_csvs, validate_csvs

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Validate/exported graph CSVs or import into an explicitly selected Neo4j database")
    p.add_argument("--directory", default=str(path(os.getenv("RARESEEK_PROCESSED_DIR", "RareMed-KG/data/processed")) / "neo4j_import"))
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    p.add_argument("--username", default=os.environ.get("NEO4J_USERNAME", "neo4j"))
    p.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    a = p.parse_args()
    if a.validate_only:
        nodes, edges = validate_csvs(path(a.directory))
        print(json.dumps({"nodes": {k: len(v) for k,v in nodes.items()}, "edges": {k: len(v) for k,v in edges.items()}}, indent=2))
    else:
        print(json.dumps(import_csvs(path(a.directory), a.uri, a.username, a.database), indent=2))
