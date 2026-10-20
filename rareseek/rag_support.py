"""Evidence-preserving helpers for three-hop retrieval workflow."""
import json
import re

def subgraph_key(subgraph):
    return json.dumps({"nodes": subgraph.get("nodes", []),
                       "relationships": subgraph.get("relationships", [])}, sort_keys=True, ensure_ascii=False)

def unique_subgraphs(subgraphs):
    unique = {subgraph_key(s): s for s in subgraphs}
    return [unique[k] for k in sorted(unique)]

def checked_paths(rows, cap):
    if len(rows) > cap:
        raise RuntimeError(f"Retrieval exceeds {cap} paths before IC ranking; raise --path-limit explicitly. No truncated ranking produced.")
    return unique_subgraphs(rows)

def variant_matches(name, transcript, change):
    """Match complete colon-delimited HGVS tokens; preserve supplied versions."""
    tokens = str(name).split(":")
    if change not in tokens:
        return False
    if "." in transcript:
        return transcript in tokens
    return any(re.fullmatch(re.escape(transcript) + r"(?:\.\d+)?", token) for token in tokens)

def positive_hpos(record):
    # No regex mining of narrative: a negated mention is not a positive phenotype.
    if "hpo_list" not in record and "hpos" not in record:
        raise ValueError("Supply an explicit positive hpo_list (use [] if none)")
    terms = record.get("hpo_list", record.get("hpos"))
    if not isinstance(terms, list) or any(not isinstance(h, str) or not re.fullmatch(r"HP:\d{7}", h) for h in terms):
        raise ValueError("Positive hpo_list must contain HP: followed by seven digits")
    return sorted(set(terms))


VARIANT_FIELDS = {"Gene.refGene", "AAChange.refGene", "hgvs.c", "Chr", "Start", "End", "Ref", "Alt",
                  "chr", "chrom", "pos", "ref", "alt"}


def patient_variants(record):
    variants = record.get('variants', [])
    if not isinstance(variants, list) or any(not isinstance(v, dict) for v in variants):
        raise ValueError('variants must be a list of objects')
    return [{k: v for k, v in variant.items() if k in VARIANT_FIELDS} for variant in variants]


def allocate_evidence(variant_groups, gene_groups, phenotypes, budget=300):
    """Strict path priority: variants, then genes, then descending-IC phenotypes."""
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    categories = {
        "variant": [r for group in variant_groups for r in group if "path" in r],
        "gene": [r for group in gene_groups for r in group if "path" in r],
        "phenotype": sorted([r for r in phenotypes if "path" in r],
                            key=lambda r: (-r["ic_score"], subgraph_key(r["path"])))
    }
    selected = []
    for rows in categories.values():
        selected.extend(rows[:max(0, budget - len(selected))])
    available = {k: len(v) for k, v in categories.items()}
    counts = {k: sum(r["kind"] == k for r in selected) for k in categories}
    unmatched = [{"kind": r["kind"], "target": r.get("target"), "status": r.get("status")}
                 for group in variant_groups + gene_groups for r in group if "path" not in r]
    return selected, {"budget": budget, "unit": "path", "priority": ["variant", "gene", "phenotype"],
                      "selected": counts, "available": available,
                      "total": len(selected), "variant_targets": len(variant_groups),
                      "gene_targets": len(gene_groups), "unmatched_targets": unmatched,
                      "omitted_by_budget": {k: available[k] - counts[k] for k in categories}}


def render_evidence(rows, terminology=None):
    """One JSON record per line. Keep directed relationship evidence visible."""
    terminology = terminology or {}
    def node(properties):
        result = {k: properties[k] for k in ("ORPHAcode", "disease_name", "HPOId", "Term",
                                            "Symbol", "variant_name", "Source", "chr", "pos", "ref", "alt", "annotations") if k in properties}
        if isinstance(result.get('annotations'), str):
            result['annotations'] = json.loads(result['annotations'])
        if result.get('HPOId') and not result.get('Term'):
            result['Term'] = terminology.get(result['HPOId'], result['HPOId'])
        return result
    def endpoint(properties):
        return {k: properties[k] for k in ('ORPHAcode', 'HPOId', 'Symbol', 'variant_name') if k in properties}
    sections = []
    for kind in ("variant", "gene", "phenotype"):
        lines = []
        for row in rows:
            if row["kind"] != kind:
                continue
            entry = {k: v for k, v in row.items() if k not in {"path", "input"}}
            if "input" in row:
                entry["input"] = {k: v for k, v in row["input"].items() if k in VARIANT_FIELDS}
            if "path" in row:
                graph = row["path"]
                entry["path_nodes"] = [node(n.get("properties", {})) for n in graph.get("nodes", [])]
                entry["edges"] = [{"type": r["type"], "source": endpoint(r.get("source", {})),
                                  "target": endpoint(r.get("target", {})),
                                  "properties": r.get('properties', {})} for r in graph.get("relationships", [])]
            lines.append(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
        sections.append(f"--- {kind.upper()} retrieval ---\n" + ("\n".join(lines) or "No paths included in this section (empty retrieval or budget exhausted)."))
    return ("Paths are retrieved evidence, not proof of causation. Non-patient phenotypes on paths "
            "must not be treated as patient findings. IC ranks path informativeness, not diagnostic probability.\n"
            + "\n\n".join(sections))
