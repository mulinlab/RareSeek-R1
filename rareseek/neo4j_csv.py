"""Validate and import graph CSVs without APOC or server file access."""
import csv
import os
import json
from .io import digest
from pathlib import Path

NODES = {"diseases": ("Disease", "ORPHAcode"), "phenotypes": ("Phenotype", "HPOId"),
         "genes": ("Gene", "Symbol"), "variants": ("Variant", "variant_name")}
EDGES = {"disease_parent": ("diseases", "diseases"), "disease_phenotype": ("diseases", "phenotypes"),
         "disease_gene": ("diseases", "genes"), "gene_variant": ("genes", "variants"),
         "gene_disease": ("genes", "diseases"), "gene_phenotype": ("genes", "phenotypes"),
         "variant_phenotype": ("variants", "phenotypes"), "variant_disease": ("variants", "diseases")}
RELATIONS = {"HAS_PARENT", "HAS_PHENOTYPE", "IS_ASSOCIATED_WITH", "HAS_VARIANT",
             "ASSOCIATED_WITH_DISEASE", "ASSOCIATED_WITH_PHENOTYPE"}

EXPECTED_TYPES = dict(zip(EDGES, ["HAS_PARENT", "HAS_PHENOTYPE", "IS_ASSOCIATED_WITH", "HAS_VARIANT", "ASSOCIATED_WITH_DISEASE", "ASSOCIATED_WITH_PHENOTYPE", "IS_ASSOCIATED_WITH", "ASSOCIATED_WITH_DISEASE"]))

def read_table(filename, required=()):
    with Path(filename).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames and len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'Duplicate CSV columns in {filename}')
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns in {filename}: {required}")
        rows = list(reader)
        if any(None in r or any(v is None for v in r.values()) for r in rows):
            raise ValueError(f"Malformed CSV row in {filename}")
        return rows

def validate_csvs(directory, check_manifest=True):
    directory = Path(directory)
    manifest = directory / 'build_manifest.json'
    if check_manifest and manifest.exists():
        info = json.loads(manifest.read_text(encoding='utf-8'))
        if info.get('status') != 'complete':
            raise ValueError('Graph construction did not complete')
        for name, checksum in info.get('csv_sha256', {}).items():
            if Path(name).name != name or digest(directory / name) != checksum:
                raise ValueError(f'Graph CSV differs from completed build: {name}')
    nodes, edges = {}, {}
    ids = {}
    for stem, (label, key) in NODES.items():
        rows = read_table(directory / f"{stem}.csv", [key + ":ID"])
        values = [r.get(key + ":ID", "") for r in rows]
        if any(not v or v != v.strip() for v in values) or len(set(values)) != len(values):
            raise ValueError(f"Missing/duplicate ID in {stem}")
        nodes[stem], ids[stem] = rows, set(values)
    if not nodes["diseases"] or not nodes["phenotypes"]:
        raise ValueError("Graph requires diseases and phenotypes")
    for stem, (start, end) in EDGES.items():
        rows = read_table(directory / f"{stem}.csv", [":START_ID", ":END_ID", "TYPE"])
        for r in rows:
            if r[":START_ID"] not in ids[start] or r[":END_ID"] not in ids[end] or r["TYPE"] != EXPECTED_TYPES[stem]:
                raise ValueError(f"Invalid edge in {stem}: {r}")
        unique = {}
        for row in rows:
            pair = (row[':START_ID'], row[':END_ID'])
            if pair in unique and unique[pair] != row:
                raise ValueError(f'Conflicting duplicate relationship in {stem}: {pair}')
            unique[pair] = row
        edges[stem] = list(unique.values())
    # A classification hierarchy must be acyclic, including self-parent links.
    remaining = {identifier: 0 for identifier in ids['diseases']}
    children = {}
    for row in edges['disease_parent']:
        child, parent = row[':START_ID'], row[':END_ID']
        remaining[child] += 1
        children.setdefault(parent, []).append(child)
    queue = [identifier for identifier, count in remaining.items() if count == 0]
    visited = 0
    while queue:
        parent = queue.pop()
        visited += 1
        for child in children.get(parent, []):
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)
    if visited != len(remaining):
        raise ValueError('Disease parent relationships contain a cycle')
    return nodes, edges

def import_csvs(directory, uri="bolt://localhost:7687", username="neo4j", database="neo4j"):
    from neo4j import GraphDatabase
    nodes, edges = validate_csvs(directory)
    if not os.environ.get("NEO4J_PASSWORD"):
        raise ValueError("NEO4J_PASSWORD is required")
    with GraphDatabase.driver(uri, auth=(username, os.environ["NEO4J_PASSWORD"])) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            if session.run("MATCH (n) RETURN count(n) AS n").single()["n"]:
                raise ValueError("Import requires an empty dedicated database; existing graph would contaminate this snapshot")
            for stem, (label, key) in NODES.items():
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE").consume()
                for start in range(0, len(nodes[stem]), 1000):
                    rows = [{k.removesuffix(":ID"): v for k, v in row.items()} for row in nodes[stem][start:start+1000]]
                    session.run(f"UNWIND $rows AS row MERGE (n:{label} {{{key}:row.{key}}}) SET n += row", rows=rows).consume()
            for stem, (start, end) in EDGES.items():
                sl, sk = NODES[start]
                el, ek = NODES[end]
                for relationship in sorted({r["TYPE"] for r in edges[stem]}):
                    selected = [r for r in edges[stem] if r["TYPE"] == relationship]
                    for offset in range(0, len(selected), 1000):
                        rows = [{"start": r[":START_ID"], "end": r[":END_ID"], "properties": {k:v for k,v in r.items() if k not in {":START_ID", ":END_ID", "TYPE"}}} for r in selected[offset:offset+1000]]
                        session.run(f"UNWIND $rows AS row MATCH (a:{sl} {{{sk}:row.start}}), (b:{el} {{{ek}:row.end}}) "
                                    f"MERGE (a)-[r:{relationship}]->(b) SET r += row.properties", rows=rows).consume()
            for stem, (label, key) in NODES.items():
                count = session.run(f'MATCH (n:{label}) RETURN count(n) AS n').single()['n']
                if count != len(nodes[stem]):
                    raise RuntimeError(f'Imported node count mismatch: {stem}')
            for stem, (start, end) in EDGES.items():
                sl, _ = NODES[start]
                el, _ = NODES[end]
                relationship = EXPECTED_TYPES[stem]
                count = session.run(f'MATCH (a:{sl})-[r:{relationship}]->(b:{el}) RETURN count(r) AS n').single()['n']
                if count != len(edges[stem]):
                    raise RuntimeError(f'Imported relationship count mismatch: {stem}')
    return {"nodes": {k: len(v) for k, v in nodes.items()}, "edges": {k: len(v) for k, v in edges.items()}}
