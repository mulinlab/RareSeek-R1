"""Convert validated exporter CSVs to portable graph, IC input, and hierarchy."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rareseek.io import path, write_json
from rareseek.neo4j_csv import validate_csvs

def snapshot(directory, output, data_kind):
    nodes, edges = validate_csvs(directory)
    diseases = {r["ORPHAcode:ID"]: {"id": r["ORPHAcode:ID"], "name": r["disease_name"],
                 "hpo_list": set(), "genes": set(), "variants": set(),
                 "aliases": [s for s in r.get("synonyms", "").split("|") if s and s != "Not provided"]} for r in nodes["diseases"]}
    hierarchy = {d: [] for d in diseases}
    for r in edges["disease_parent"]:
        hierarchy[r[":START_ID"]].append(r[":END_ID"])
    for stem, field, reverse in (("disease_phenotype", "hpo_list", False), ("disease_gene", "genes", False),
                                  ("gene_disease", "genes", True), ("variant_disease", "variants", True)):
        for r in edges[stem]:
            did, value = (r[":END_ID"], r[":START_ID"]) if reverse else (r[":START_ID"], r[":END_ID"])
            diseases[did][field].add(value)
    for d in diseases.values():
        for field in ("hpo_list", "genes", "variants"):
            d[field] = sorted(d[field])
    write_json(output / "graph.json", {"data_kind": data_kind, "diseases": list(diseases.values())})
    write_json(output / "hierarchy.json", {k: sorted(set(v)) for k,v in hierarchy.items()})
    write_json(output / "HPO_phenotype.json", {k: {"hpo_list": v["hpo_list"]} for k,v in diseases.items()})

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data-kind", choices=["private", "public"], required=True)
    a = p.parse_args()
    snapshot(path(a.directory), path(a.output), a.data_kind)
