from __future__ import annotations

import csv
import gzip
import json
import logging
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path("/mnt/hpc/home/scllm/deepseek_model/rare_disease/rare_disease")
COMPLETE_DIR = Path("/mnt/hpc/home/scllm/deepseek_model/rare_disease/new/complete")
ORPHA_DIR = COMPLETE_DIR / "Orphapacket"
JSONKG_DIR = COMPLETE_DIR / "json_KG"
CLINVAR_DIR = COMPLETE_DIR / "clinvar"
OUT_DIR = COMPLETE_DIR / "neo4j_import"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_CACHE_PATH = CLINVAR_DIR / "cache" / "variant_restructured.pkl.gz"

LOGGER = logging.getLogger("build_neo4j_import")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PLACEHOLDER_VALUES = {"", ".", "-", "Unknown", "unknown", "None", "null", "N/A", "na"}


def is_valid_value(v) -> bool:
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return len(v) > 0
    if pd.isna(v):
        return False
    return str(v).strip() not in PLACEHOLDER_VALUES


def normalize_text(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return str(v).strip()


def split_tokens(value: str, pattern: str) -> List[str]:
    text = normalize_text(value)
    if not text:
        return []
    return [item.strip() for item in re.split(pattern, text) if item and item.strip()]


def normalize_orpha_code(v) -> str:
    text = normalize_text(v)
    if not text:
        return ""
    upper = text.upper()
    if upper.startswith("ORPHA:"):
        suffix = text.split(":", 1)[1].strip()
        return f"ORPHA:{suffix}" if suffix else ""
    if re.fullmatch(r"\d+", text):
        return f"ORPHA:{text}"
    return text


def normalize_hpo_id(v) -> str:
    text = normalize_text(v)
    if not text:
        return ""
    match = re.search(r"(HP:\d{7})", text.upper())
    if match:
        return match.group(1)
    if re.fullmatch(r"HP:\d{7}", text.upper()):
        return text.upper()
    return ""


def normalize_reference_id(value) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    match = re.search(r"(?i)\b(ORPHA|ORPHANET|OMIM)\s*:\s*([A-Za-z0-9]+)", text)
    if not match:
        return text
    prefix = match.group(1).upper()
    suffix = match.group(2).strip()
    if prefix == "ORPHANET":
        prefix = "ORPHA"
    return f"{prefix}:{suffix}"


def normalize_gene_symbol(v) -> str:
    text = normalize_text(v)
    if not text:
        return ""
    if text in PLACEHOLDER_VALUES or text in {"-", "."}:
        return ""
    if re.fullmatch(r"[\-.]+", text):
        return ""
    return text


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def dedupe_preserve_order(rows: Iterable[dict], key_fields: Sequence[str]) -> List[dict]:
    seen = set()
    out = []
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def load_pickle_cache(path: Path):
    if not path.exists():
        return None
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)

def export_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    LOGGER.info("exported %s", path)


def load_orphapacket_master() -> List[dict]:
    LOGGER.info("loading Orphapacket master from %s", ORPHA_DIR)
    disease_df = read_tsv(ORPHA_DIR / "rare_disease_data_combined.tsv")
    age_df = read_csv(ORPHA_DIR / "AverageAgeOfOnsets.csv")
    inheritance_df = read_csv(ORPHA_DIR / "TypeOfInheritances.csv")
    if "Label" in inheritance_df.columns:
        inheritance_df = inheritance_df.rename(columns={"Label": "InheritanceLabel"})
    phenotype_df = read_tsv(ORPHA_DIR / "phenotypes_extracted.tsv")
    gene_df = read_tsv(ORPHA_DIR / "genes_extracted.tsv")
    parents_df = read_tsv(ORPHA_DIR / "parents_extracted.tsv")

    for df in [disease_df, age_df, inheritance_df, phenotype_df, gene_df, parents_df]:
        if "ORPHAcode" in df.columns:
            df["ORPHAcode"] = df["ORPHAcode"].map(normalize_orpha_code)

    merged_df = disease_df.copy()
    for extra in [age_df, inheritance_df, phenotype_df, gene_df, parents_df]:
        if "ORPHAcode" in extra.columns:
            extra = extra.copy()
            if "Disease" in extra.columns:
                extra = extra.drop(columns=["Disease"])
            merged_df = pd.merge(merged_df, extra, on="ORPHAcode", how="left")

    merged_df = merged_df.fillna("")
    records: List[dict] = []
    for _, row in merged_df.iterrows():
        orpha = normalize_orpha_code(row.get("ORPHAcode"))
        if not orpha:
            continue

        disease_name = normalize_text(row.get("Label")) or normalize_text(row.get("Label_x")) or normalize_text(row.get("Label_y")) or f"Unknown_ORPHA_{row.get('ORPHAcode')}"
        average_age = [
            normalize_text(row.get(f"AverageAgeOfOnset_value{i}", ""))
            for i in range(1, 7)
            if is_valid_value(row.get(f"AverageAgeOfOnset_value{i}"))
        ]
        inheritance_types = [
            normalize_text(row.get(f"TypeOfInheritance_value{i}", ""))
            for i in range(1, 7)
            if is_valid_value(row.get(f"TypeOfInheritance_value{i}"))
        ]

        phenotypes = []
        for i in range(1, 189):
            key = f"phenotype_HPOId{i}"
            if key in row and is_valid_value(row.get(key)):
                ph_id = normalize_hpo_id(row.get(key))
                if ph_id:
                    phenotypes.append(
                        {
                            "HPOId": ph_id,
                            "Term": normalize_text(row.get(f"phenotype_HPOTerm{i}", "")),
                            "Frequency": normalize_text(row.get(f"phenotype_HPOFrequency{i}", "")),
                        }
                    )

        genes = []
        for i in range(1, 109):
            key = f"gene_Symbol{i}"
            if key in row and is_valid_value(row.get(key)):
                symbol = normalize_gene_symbol(row.get(key))
                if symbol:
                    genes.append(
                        {
                            "Symbol": symbol,
                            "Name": normalize_text(row.get(f"gene_Name{i}", "")),
                            "ExternalReferences": normalize_text(row.get(f"gene_ExternalReferences{i}", "")),
                            "AssociationType": normalize_text(row.get(f"gene_DisorderGeneAssociationType{i}", "")),
                        }
                    )

        parents = []
        for i in range(1, 31):
            key = f"parent_ORPHAcode{i}"
            if key in row and is_valid_value(row.get(key)):
                pid = normalize_orpha_code(row.get(key))
                if pid:
                    parents.append({"ORPHAcode": pid, "Label": normalize_text(row.get(f"parent_Label{i}", ""))})

        records.append(
            {
                "ORPHAcode": orpha,
                "properties": {
                    "disease_name": disease_name,
                    "synonyms": normalize_text(row.get("Synonyms", "")),
                    "externalReferences": normalize_text(row.get("ExternalReferences", "")),
                    "averageAgeOfOnset": average_age,
                    "inheritanceTypes": inheritance_types,
                },
                "phenotypes": phenotypes,
                "genes": genes,
                "parents": parents,
            }
        )

    LOGGER.info("orpha records: %s", len(records))
    return records


def load_disease_gene_supplement() -> List[dict]:
    path = COMPLETE_DIR / 'data' / 'phenotype_to_genes_20210413.txt'
    LOGGER.info("loading disease-gene supplement: %s", path)
    column_names = [
        "HPO-id",
        "HPO-label",
        "entrez-gene-id",
        "entrez-gene-symbol",
        "Additional-Info",
        "G-D-source",
        "disease-ID",
    ]
    df = pd.read_csv(path, sep="	", header=0, names=column_names, dtype=str, keep_default_na=False, encoding="utf-8")
    out = []
    for _, row in df.iterrows():
        raw_disease = normalize_text(row.get("disease-ID"))
        disease_id = normalize_reference_id(raw_disease)
        if not disease_id and re.fullmatch(r"\d+", raw_disease):
            disease_id = f"ORPHA:{raw_disease}"
        gene_symbol = normalize_gene_symbol(row.get("entrez-gene-symbol"))
        phenotype_id = normalize_hpo_id(row.get("HPO-id"))
        phenotype_term = normalize_text(row.get("HPO-label"))
        if not (disease_id or gene_symbol or phenotype_id):
            continue
        out.append(
            {
                "disease": {"ORPHAcode": disease_id},
                "phenotype": {"HPOId": phenotype_id, "Term": phenotype_term},
                "gene": {
                    "Symbol": gene_symbol,
                    "properties": {
                        "HPO-id": phenotype_id,
                        "HPO-label": phenotype_term,
                        "entrez-gene-id": normalize_text(row.get("entrez-gene-id")),
                        "G-D-source": normalize_text(row.get("G-D-source")),
                    },
                },
            }
        )
    LOGGER.info("disease-gene supplement rows: %s", len(out))
    return out


def load_orphapacket_gene_relations() -> Tuple[OrderedDict, set, set]:
    """
    Load Orphapacket gene nodes and disease-parent relations.

    Returns:
        gene_nodes: ordered gene node properties keyed by Symbol
        disease_gene_rels: (disease ORPHA, gene symbol)
        disease_parent_rels: (disease ORPHA, parent ORPHA)
    """
    gene_nodes: OrderedDict[str, dict] = OrderedDict()
    disease_gene_rels: set[tuple[str, str]] = set()
    disease_parent_rels: set[tuple[str, str]] = set()

    gene_path = ORPHA_DIR / 'genes_extracted.tsv'
    if gene_path.exists():
        df = read_tsv(gene_path)
        for _, row in df.iterrows():
            disease = normalize_orpha_code(row.get('ORPHAcode'))
            if not disease:
                continue
            for i in range(1, 109):
                symbol = normalize_gene_symbol(row.get(f'gene_Symbol{i}', ''))
                if not symbol:
                    continue
                gene_nodes.setdefault(
                    symbol,
                    {
                        'Name': normalize_text(row.get(f'gene_Name{i}', '')),
                        'ExternalReferences': normalize_text(row.get(f'gene_ExternalReferences{i}', '')),
                        'AssociationType': normalize_text(row.get(f'gene_DisorderGeneAssociationType{i}', '')),
                        'HPO-id': '',
                        'HPO-label': '',
                        'entrez-gene-id': '',
                        'G-D-source': '',
                    },
                )
                disease_gene_rels.add((disease, symbol))

    parent_path = ORPHA_DIR / 'parents_extracted.tsv'
    if parent_path.exists():
        df = read_tsv(parent_path)
        for _, row in df.iterrows():
            disease = normalize_orpha_code(row.get('ORPHAcode'))
            if not disease:
                continue
            for i in range(1, 16):
                parent = normalize_orpha_code(row.get(f'parent_ORPHAcode{i}', ''))
                if parent:
                    disease_parent_rels.add((disease, parent))

    LOGGER.info(
        'orphapacket gene nodes=%s, disease-gene rels=%s, disease-parent rels=%s',
        len(gene_nodes),
        len(disease_gene_rels),
        len(disease_parent_rels),
    )
    return gene_nodes, disease_gene_rels, disease_parent_rels
def load_json_kg_relation_sets() -> Tuple[set, set, set, set]:
    gene_disease_rels = set()
    gene_phenotype_rels = set()
    phenotype_disease_rels = set()
    gene_symbols = set()

    candidate_files = [
        JSONKG_DIR / "gene_disease_kg_extracted.csv",
        JSONKG_DIR / "gene_phenotype_kg_extracted.csv",
        JSONKG_DIR / "hpo_phenotype_kg_extracted.csv",
        JSONKG_DIR / "phenotype_to_genes_kg_extracted.csv",
    ]

    for path in candidate_files:
        if not path.exists():
            continue
        df = read_csv(path)
        basename = path.name
        for _, row in df.iterrows():
            disease_id = normalize_reference_id(
                row.get("disease_id") or row.get("disease") or row.get("disease-id") or row.get("diseaseid")
            )
            if disease_id.startswith('DECIPHER:'):
                continue
            gene_symbol = normalize_gene_symbol(
                row.get("gene") or row.get("gene_symbol") or row.get("entrez-gene-symbol") or row.get("disease")
            )
            hpo_id = normalize_hpo_id(row.get("hpo_id") or row.get("HPOId") or row.get("hpo"))
            if gene_symbol:
                gene_symbols.add(gene_symbol)

            if basename == "gene_disease_kg_extracted.csv":
                if gene_symbol and disease_id:
                    gene_disease_rels.add((gene_symbol, disease_id))
                continue

            if basename == "gene_phenotype_kg_extracted.csv":
                if gene_symbol and hpo_id:
                    gene_phenotype_rels.add((gene_symbol, hpo_id))
                if gene_symbol and disease_id:
                    gene_disease_rels.add((gene_symbol, disease_id))
                continue

            if basename == "hpo_phenotype_kg_extracted.csv":
                if hpo_id and disease_id:
                    phenotype_disease_rels.add((hpo_id, disease_id))
                continue

            if basename == "phenotype_to_genes_kg_extracted.csv":
                if gene_symbol and disease_id:
                    gene_disease_rels.add((gene_symbol, disease_id))
                if gene_symbol and hpo_id:
                    gene_phenotype_rels.add((gene_symbol, hpo_id))
                if hpo_id and disease_id:
                    phenotype_disease_rels.add((hpo_id, disease_id))

    LOGGER.info("json kg gene symbols: %s", len(gene_symbols))
    return gene_disease_rels, gene_phenotype_rels, phenotype_disease_rels, gene_symbols


def load_json_kg_disease_rows() -> List[dict]:
    rows = []
    seen = set()
    candidate_files = [
        JSONKG_DIR / "gene_disease_kg_extracted.csv",
    ]
    for path in candidate_files:
        if not path.exists():
            continue
        df = read_csv(path)
        for _, row in df.iterrows():
            disease_id = normalize_reference_id(
                row.get("disease_id") or row.get("disease") or row.get("disease-id") or row.get("diseaseid")
            )
            disease_name = normalize_text(row.get("disease_name"))
            if not disease_id or disease_id in seen:
                continue
            seen.add(disease_id)
            rows.append(
                {
                    "ORPHAcode:ID": disease_id,
                    "disease_name": disease_name or disease_id,
                    "synonyms": "",
                    "externalReferences": "",
                    "averageAgeOfOnset": "",
                    "inheritanceTypes": "",
                }
            )
    LOGGER.info("json kg disease rows: %s", len(rows))
    return rows


def load_json_kg_phenotype_rows() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    candidate_files = [
        JSONKG_DIR / "hpo_phenotype_kg_extracted.csv",
        JSONKG_DIR / "gene_phenotype_kg_extracted.csv",
        JSONKG_DIR / "phenotype_to_genes_kg_extracted.csv",
    ]
    for path in candidate_files:
        if not path.exists():
            continue
        df = read_csv(path)
        for _, row in df.iterrows():
            hpo_id = normalize_hpo_id(row.get("hpo_id") or row.get("HPOId") or row.get("hpo"))
            hpo_name = normalize_text(row.get("hpo_name") or row.get("hpo_term") or row.get("chpo_term"))
            if hpo_id:
                out.setdefault(hpo_id, {"Term": hpo_name, "Frequency": ""})
    LOGGER.info("json kg phenotype nodes: %s", len(out))
    return out


def load_clinvar_outputs():
    cache_path = CLINVAR_DIR / "cache" / "variant_restructured.pkl.gz"
    cached = load_pickle_cache(cache_path)
    if cached is None:
        raise FileNotFoundError(f"missing ClinVar cache: {cache_path}")

    if len(cached) != 7:
        raise ValueError(f"unexpected ClinVar cache shape: {len(cached)}")

    variant_nodes, variant_phenotype_rels, variant_disease_rels, gene_variant_rels, _, _, phenotype_disease_rels = cached

    def load_edge_set(filename: str) -> set[tuple[str, str]]:
        path = CLINVAR_DIR / filename
        if not path.exists():
            return set()
        df = read_csv(path)
        return {
            (normalize_text(r.get(":START_ID")), normalize_text(r.get(":END_ID")))
            for _, r in df.iterrows()
            if normalize_text(r.get(":START_ID")) and normalize_text(r.get(":END_ID"))
        }

    gene_disease_rels = load_edge_set("gene_disease.csv")
    gene_phenotype_rels = load_edge_set("gene_phenotype.csv")

    LOGGER.info(
        "loaded clinvar cache: variant_nodes=%s variant_phenotype=%s variant_disease=%s gene_variant=%s phenotype_disease=%s gene_disease_csv=%s gene_phenotype_csv=%s",
        len(variant_nodes),
        len(variant_phenotype_rels),
        len(variant_disease_rels),
        len(gene_variant_rels),
        len(phenotype_disease_rels),
        len(gene_disease_rels),
        len(gene_phenotype_rels),
    )
    return variant_nodes, set(variant_phenotype_rels), set(variant_disease_rels), set(gene_variant_rels), gene_disease_rels, gene_phenotype_rels, set(phenotype_disease_rels)


def load_clinvar_variant_rows() -> List[dict]:
    df = read_csv(CLINVAR_DIR / "variants.csv")
    rows = []
    for _, row in df.iterrows():
        vid = normalize_text(row.get("variant_name:ID"))
        if not vid:
            continue
        rows.append({
            "variant_name:ID": vid,
            "Source": normalize_text(row.get("Source", "")),
            "chr": normalize_text(row.get("chr", "")),
            "pos": normalize_text(row.get("pos", "")),
            "ref": normalize_text(row.get("ref", "")),
            "alt": normalize_text(row.get("alt", "")),
            "consequence_effect": normalize_text(row.get("consequence_effect", "")),
            "consequence_protein_id": normalize_text(row.get("consequence_protein_id", "")),
            "dbsnp": normalize_text(row.get("dbsnp", "")),
            "revel_score": normalize_text(row.get("revel_score", "")),
            "revel_prediction": normalize_text(row.get("revel_prediction", "")),
            "spliceai_max_score": normalize_text(row.get("spliceai_max_score", "")),
            "spliceai_max_prediction": normalize_text(row.get("spliceai_max_prediction", "")),
            "acmg_score": normalize_text(row.get("acmg_score", "")),
            "acmg_criteria": normalize_text(row.get("acmg_criteria", "")),
            "clinvar_classification": normalize_text(row.get("clinvar_classification", "")),
            "consequence_gene_symbol": normalize_text(row.get("consequence_gene_symbol", "")),
        })
    return dedupe_preserve_order(rows, ["variant_name:ID"])


def load_clinvar_variant_edges(filename: str) -> List[dict]:
    path = CLINVAR_DIR / filename
    if not path.exists():
        return []
    df = read_csv(path)
    return [{":START_ID": normalize_text(r.get(":START_ID")), ":END_ID": normalize_text(r.get(":END_ID")), "TYPE": normalize_text(r.get("TYPE", ""))} for _, r in df.iterrows()]


def build_and_export():
    disease_data = load_orphapacket_master()
    disease_gene_records = load_disease_gene_supplement()
    json_gene_disease_rels, json_gene_phenotype_rels, json_phenotype_disease_rels, json_gene_symbols = load_json_kg_relation_sets()
    orpha_gene_nodes, orpha_gene_disease_rels, orpha_gene_parent_rels = load_orphapacket_gene_relations()
    (
        clinvar_variant_nodes,
        clinvar_variant_phenotype_rels,
        clinvar_variant_disease_rels,
        clinvar_gene_variant_rels,
        clinvar_gene_disease_rels,
        clinvar_gene_phenotype_rels,
        clinvar_phenotype_disease_rels,
    ) = load_clinvar_outputs()

    disease_nodes = OrderedDict()
    phenotype_nodes = OrderedDict()
    gene_nodes = OrderedDict()
    variant_nodes_out = OrderedDict(clinvar_variant_nodes)

    disease_relations_parent = set()
    disease_relations_dp = set()
    disease_relations_dg = set()
    gene_relations_disease = set()
    gene_phenotype_rels = set()
    phenotype_disease_rels = set()

    for rec in disease_data:
        did = normalize_orpha_code(rec.get("ORPHAcode"))
        if not did:
            continue
        disease_nodes[did] = rec.get("properties", {})

        for ph in rec.get("phenotypes", []):
            ph_id = normalize_hpo_id(ph.get("HPOId"))
            if ph_id:
                phenotype_nodes[ph_id] = {
                    "Term": normalize_text(ph.get("Term", "")),
                    "Frequency": normalize_text(ph.get("Frequency", "")),
                }
                disease_relations_dp.add((did, ph_id))

        for gene in rec.get("genes", []):
            gs = normalize_gene_symbol(gene.get("Symbol"))
            if gs:
                gene_nodes[gs] = {
                    "Name": normalize_text(gene.get("Name", "")),
                    "ExternalReferences": normalize_text(gene.get("ExternalReferences", "")),
                    "AssociationType": normalize_text(gene.get("AssociationType", "")),
                    "HPO-id": "",
                    "HPO-label": "",
                    "entrez-gene-id": "",
                    "G-D-source": "",
                }
                disease_relations_dg.add((did, gs))
                gene_relations_disease.add((gs, did))

        for parent in rec.get("parents", []):
            pid = normalize_orpha_code(parent.get("ORPHAcode"))
            if pid:
                if pid not in disease_nodes:
                    disease_nodes[pid] = {"disease_name": normalize_text(parent.get("Label", ""))}
                disease_relations_parent.add((did, pid))

    for rec in disease_gene_records:
        did = normalize_reference_id(rec["disease"].get("ORPHAcode"))
        gs = normalize_gene_symbol(rec["gene"].get("Symbol"))
        ph_id = normalize_hpo_id(rec.get("phenotype", {}).get("HPOId"))
        ph_term = normalize_text(rec.get("phenotype", {}).get("Term", ""))

        if did and did not in disease_nodes:
            disease_nodes[did] = {
                "disease_name": did,
                "synonyms": "Not provided",
                "externalReferences": "Not provided",
                "averageAgeOfOnset": "",
                "inheritanceTypes": "",
            }
        if ph_id:
            phenotype_nodes.setdefault(ph_id, {"Term": ph_term, "Frequency": ""})
        if gs and gs not in gene_nodes:
            gene_nodes[gs] = rec["gene"].get("properties", {})

        if did and gs:
            disease_relations_dg.add((did, gs))
            gene_relations_disease.add((gs, did))
        if gs and ph_id:
            gene_phenotype_rels.add((gs, ph_id))
        if ph_id and did:
            phenotype_disease_rels.add((ph_id, did))

    for gs in sorted(json_gene_symbols):
        if gs and gs not in gene_nodes:
            gene_nodes[gs] = {
                "Name": "",
                "ExternalReferences": "",
                "AssociationType": "",
                "HPO-id": "",
                "HPO-label": "",
                "entrez-gene-id": "",
                "G-D-source": "",
            }

    for gs, props in orpha_gene_nodes.items():
        if gs not in gene_nodes:
            gene_nodes[gs] = props

    for gs, disease in sorted(json_gene_disease_rels):
        if gs:
            gene_relations_disease.add((gs, disease))
    for gs, ph in sorted(json_gene_phenotype_rels):
        if gs and ph:
            gene_phenotype_rels.add((gs, ph))
    for ph, disease in sorted(json_phenotype_disease_rels):
        if ph and disease and normalize_hpo_id(ph) and normalize_reference_id(disease):
            phenotype_disease_rels.add((ph, disease))
            disease_relations_dp.add((disease, ph))

    for disease, parent in sorted(orpha_gene_parent_rels):
        if disease and parent:
            disease_relations_parent.add((disease, parent))

    for gs, vid in clinvar_gene_variant_rels:
        if gs and gs not in gene_nodes:
            gene_nodes[gs] = {
                "Name": "",
                "ExternalReferences": "",
                "AssociationType": "",
                "HPO-id": "",
                "HPO-label": "",
                "entrez-gene-id": "",
                "G-D-source": "",
            }
        variant_nodes_out.setdefault(vid, clinvar_variant_nodes.get(vid, {}))

    for vid, ph in clinvar_variant_phenotype_rels:
        if ph:
            phenotype_nodes.setdefault(ph, {"Term": "", "Frequency": ""})
    for vid, disease_term in clinvar_variant_disease_rels:
        if disease_term:
            normalized_disease = normalize_reference_id(disease_term)
            if normalized_disease.startswith('DECIPHER:'):
                continue
            if normalized_disease and normalized_disease.startswith('HP:'):
                continue
            if 'HP:' in disease_term and ('present' in disease_term.lower() or ';' in disease_term):
                continue
            node_id = normalized_disease if normalized_disease and (normalized_disease.startswith('OMIM:') or normalized_disease.startswith('ORPHA:') or normalized_disease.startswith('DECIPHER:')) else disease_term
            disease_nodes.setdefault(
                node_id,
                {
                    'disease_name': node_id,
                    'synonyms': 'Not provided',
                    'externalReferences': 'Not provided',
                    'averageAgeOfOnset': '',
                    'inheritanceTypes': '',
                },
            )

    for gs, disease_term in clinvar_gene_disease_rels:
        if gs and disease_term:
            normalized_disease = normalize_reference_id(disease_term)
            if normalized_disease and normalized_disease.startswith("HP:"):
                continue
            gene_relations_disease.add((gs, disease_term))

    for gs, ph in clinvar_gene_phenotype_rels:
        if gs and ph:
            gene_phenotype_rels.add((gs, ph))
    for ph, disease_term in clinvar_phenotype_disease_rels:
        if ph and disease_term and normalize_hpo_id(ph) and normalize_reference_id(disease_term):
            normalized_disease = normalize_reference_id(disease_term)
            if normalized_disease and normalized_disease.startswith("HP:"):
                continue
            phenotype_disease_rels.add((ph, disease_term))

    for row in load_json_kg_disease_rows():
        did = row["ORPHAcode:ID"]
        if did not in disease_nodes:
            disease_nodes[did] = {
                "disease_name": row["disease_name"],
                "synonyms": row["synonyms"],
                "externalReferences": row["externalReferences"],
                "averageAgeOfOnset": row["averageAgeOfOnset"],
                "inheritanceTypes": row["inheritanceTypes"],
            }

    for ph_id, props in load_json_kg_phenotype_rows().items():
        phenotype_nodes.setdefault(ph_id, props)

    vp_rows = [
        {":START_ID": vid, ":END_ID": ph, "TYPE": "IS_ASSOCIATED_WITH"}
        for vid, ph in sorted(clinvar_variant_phenotype_rels)
        if vid in variant_nodes_out and ph in phenotype_nodes
    ]
    vd_rows = [
        {":START_ID": vid, ":END_ID": disease_term, "TYPE": "ASSOCIATED_WITH_DISEASE"}
        for vid, disease_term in sorted(clinvar_variant_disease_rels)
        if vid in variant_nodes_out and disease_term in disease_nodes
    ]
    disease_rows = [
        {
            "ORPHAcode:ID": did,
            "disease_name": normalize_text(props.get("disease_name", "")) or did,
            "synonyms": normalize_text(props.get("synonyms", "")),
            "externalReferences": normalize_text(props.get("externalReferences", "")),
            "averageAgeOfOnset": "|".join(props.get("averageAgeOfOnset", [])) if isinstance(props.get("averageAgeOfOnset"), list) else normalize_text(props.get("averageAgeOfOnset", "")),
            "inheritanceTypes": "|".join(props.get("inheritanceTypes", [])) if isinstance(props.get("inheritanceTypes"), list) else normalize_text(props.get("inheritanceTypes", "")),
        }
        for did, props in disease_nodes.items()
    ]
    disease_rows = dedupe_preserve_order(disease_rows, ["ORPHAcode:ID"])

    phenotype_rows = [
        {
            "HPOId:ID": ph_id,
            "Term": normalize_text(props.get("Term", "")),
            "Frequency": normalize_text(props.get("Frequency", "")),
        }
        for ph_id, props in phenotype_nodes.items()
        if ph_id
    ]
    phenotype_rows = dedupe_preserve_order(phenotype_rows, ["HPOId:ID"])

    gene_rows = [
        {
            "Symbol:ID": gs,
            "Name": normalize_text(props.get("Name", "")),
            "ExternalReferences": normalize_text(props.get("ExternalReferences", "")),
            "AssociationType": normalize_text(props.get("AssociationType", "")),
            "HPO-id": normalize_text(props.get("HPO-id", "")),
            "HPO-label": normalize_text(props.get("HPO-label", "")),
            "entrez-gene-id": normalize_text(props.get("entrez-gene-id", "")),
            "G-D-source": normalize_text(props.get("G-D-source", "")),
        }
        for gs, props in gene_nodes.items()
    ]
    gene_rows = dedupe_preserve_order(gene_rows, ["Symbol:ID"])

    variant_rows = [
        {
            "variant_name:ID": vid,
            "Source": normalize_text(props.get("Source", "")),
            "chr": normalize_text(props.get("chr", "")),
            "pos": normalize_text(props.get("pos", "")),
            "ref": normalize_text(props.get("ref", "")),
            "alt": normalize_text(props.get("alt", "")),
            "consequence_effect": normalize_text(props.get("consequence_effect", "")),
            "consequence_protein_id": normalize_text(props.get("consequence_protein_id", "")),
            "dbsnp": normalize_text(props.get("dbsnp", "")),
            "revel_score": normalize_text(props.get("revel_score", "")),
            "revel_prediction": normalize_text(props.get("revel_prediction", "")),
            "spliceai_max_score": normalize_text(props.get("spliceai_max_score", "")),
            "spliceai_max_prediction": normalize_text(props.get("spliceai_max_prediction", "")),
            "acmg_score": normalize_text(props.get("acmg_score", "")),
            "acmg_criteria": normalize_text(props.get("acmg_criteria", "")),
            "clinvar_classification": normalize_text(props.get("clinvar_classification", "")),
            "consequence_gene_symbol": normalize_text(props.get("consequence_gene_symbol", "")),
        }
        for vid, props in variant_nodes_out.items()
    ]
    variant_rows = dedupe_preserve_order(variant_rows, ["variant_name:ID"])

    export_csv(
        OUT_DIR / "diseases.csv",
        ["ORPHAcode:ID", "disease_name", "synonyms", "externalReferences", "averageAgeOfOnset", "inheritanceTypes"],
        disease_rows,
    )
    export_csv(OUT_DIR / "phenotypes.csv", ["HPOId:ID", "Term", "Frequency"], phenotype_rows)
    export_csv(
        OUT_DIR / "genes.csv",
        ["Symbol:ID", "Name", "ExternalReferences", "AssociationType", "HPO-id", "HPO-label", "entrez-gene-id", "G-D-source"],
        gene_rows,
    )
    export_csv(
        OUT_DIR / "variants.csv",
        [
            "variant_name:ID",
            "Source",
            "chr",
            "pos",
            "ref",
            "alt",
            "consequence_effect",
            "consequence_protein_id",
            "dbsnp",
            "revel_score",
            "revel_prediction",
            "spliceai_max_score",
            "spliceai_max_prediction",
            "acmg_score",
            "acmg_criteria",
            "clinvar_classification",
            "consequence_gene_symbol",
        ],
        variant_rows,
    )

    export_csv(
        OUT_DIR / "disease_parent.csv",
        [":START_ID", ":END_ID", "TYPE"],
        [{":START_ID": d, ":END_ID": p, "TYPE": "HAS_PARENT"} for d, p in sorted(disease_relations_parent) if d in disease_nodes and p in disease_nodes],
    )
    export_csv(
        OUT_DIR / "disease_phenotype.csv",
        [":START_ID", ":END_ID", "TYPE"],
        [{":START_ID": d, ":END_ID": ph, "TYPE": "HAS_PHENOTYPE"} for d, ph in sorted(disease_relations_dp) if d in disease_nodes and ph in phenotype_nodes],
    )
    export_csv(
        OUT_DIR / "disease_gene.csv",
        [":START_ID", ":END_ID", "TYPE"],
        [{":START_ID": d, ":END_ID": gs, "TYPE": "IS_ASSOCIATED_WITH"} for d, gs in sorted(disease_relations_dg) if d in disease_nodes and gs in gene_nodes],
    )
    export_csv(
        OUT_DIR / "gene_variant.csv",
        [":START_ID", ":END_ID", "TYPE"],
        [{":START_ID": gs, ":END_ID": vid, "TYPE": "HAS_VARIANT"} for gs, vid in sorted(clinvar_gene_variant_rels) if gs in gene_nodes and vid in variant_nodes_out],
    )
    export_csv(
        OUT_DIR / "gene_disease.csv",
        [":START_ID", ":END_ID", "TYPE"],
        [{":START_ID": gs, ":END_ID": disease, "TYPE": "ASSOCIATED_WITH_DISEASE"} for gs, disease in sorted(gene_relations_disease) if gs in gene_nodes and disease in disease_nodes],
    )
    export_csv(
        OUT_DIR / "gene_phenotype.csv",
        [":START_ID", ":END_ID", "TYPE"],
        [{":START_ID": gs, ":END_ID": ph, "TYPE": "ASSOCIATED_WITH_PHENOTYPE"} for gs, ph in sorted(gene_phenotype_rels) if gs in gene_nodes and ph in phenotype_nodes],
    )
    export_csv(
        OUT_DIR / "variant_phenotype.csv",
        [":START_ID", ":END_ID", "TYPE"],
        vp_rows,
    )
    export_csv(
        OUT_DIR / "variant_disease.csv",
        [":START_ID", ":END_ID", "TYPE"],
        vd_rows,
    )

    LOGGER.info(
        "final counts: diseases=%s phenotypes=%s genes=%s variants=%s",
        len(disease_rows),
        len(phenotype_rows),
        len(gene_rows),
        len(variant_rows),
    )
    LOGGER.info(
        "final relations: disease_parent=%s disease_gene=%s gene_disease=%s gene_phenotype=%s phenotype_disease=%s gene_variant=%s variant_disease=%s variant_phenotype=%s",
        len(disease_relations_parent),
        len(disease_relations_dg),
        len(gene_relations_disease),
        len(gene_phenotype_rels),
        len(phenotype_disease_rels),
        len(clinvar_gene_variant_rels),
        len(clinvar_variant_disease_rels),
        len(clinvar_variant_phenotype_rels),
    )


def main():
    build_and_export()


if __name__ == "__main__":
    main()


