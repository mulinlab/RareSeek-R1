from __future__ import annotations

import csv
import gzip
import logging
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path("..data/raw")
OUT_DIR = Path("../processed/variant")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
VARIANT_CACHE_PATH = CACHE_DIR / "variant_restructured.pkl.gz"

VARIANT_CSV_FILE_PATH = BASE_DIR / "variants_HGMD_Clinvar.csv"

LOGGER = logging.getLogger("process_clinvar")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PLACEHOLDER_VALUES = {"", ".", "-", "Unknown", "unknown", "None", "null", "N/A", "na"}


def load_pickle_cache(path: Path):
    if not path.exists():
        return None
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)


def save_pickle_cache(path: Path, payload):
    with gzip.open(path, "wb", compresslevel=1) as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


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


def split_multi_delim(value: str) -> List[str]:
    return split_tokens(value, r"[|,;]")


def split_pipe(value: str) -> List[str]:
    return split_tokens(value, r"[|]")


def split_pipe_comma(value: str) -> List[str]:
    return split_tokens(value, r"[|,]")


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


def normalize_gene_symbol(v) -> str:
    text = normalize_text(v)
    if not text:
        return ""
    if text in PLACEHOLDER_VALUES or text in {"-", "."}:
        return ""
    if re.fullmatch(r"[\-.]+", text):
        return ""
    return text


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


def export_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    LOGGER.info("exported %s", path)


def build_variant_name(row: pd.Series, fallback_index: int) -> str:
    parts = [
        normalize_text(row.get("consequence_transcript", "")),
        normalize_text(row.get("consequence_hgvs_c", "")),
        normalize_text(row.get("consequence_hgvs_p", "")),
    ]
    parts = [p for p in parts if p]
    if parts:
        return ":".join(parts)

    chr_val = normalize_text(row.get("chr", ""))
    pos_val = normalize_text(row.get("pos", ""))
    ref_val = normalize_text(row.get("ref", ""))
    alt_val = normalize_text(row.get("alt", ""))
    if chr_val and pos_val and ref_val and alt_val:
        return f"variant:{chr_val}:{pos_val}:{ref_val}>{alt_val}"
    return f"variant_{fallback_index}"


def load_variant_data():
    LOGGER.info("loading variant data: %s", VARIANT_CSV_FILE_PATH)
    df = pd.read_csv(VARIANT_CSV_FILE_PATH, sep="	", dtype=str, keep_default_na=False, encoding="utf-8")
    LOGGER.info("variant rows: %s", len(df))

    if "variant_name" not in df.columns:
        df["variant_name"] = [build_variant_name(row, idx) for idx, (_, row) in enumerate(df.iterrows(), start=1)]

    variant_nodes = OrderedDict()
    variant_phenotype_rels = set()
    variant_disease_rels = set()
    gene_variant_rels = set()
    gene_disease_rels = set()
    gene_phenotype_rels = set()
    phenotype_disease_rels = set()

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        variant_id = normalize_text(row.get("variant_name")) or build_variant_name(row, idx)
        if not variant_id:
            continue

        variant_nodes[variant_id] = {
            "Source": normalize_text(row.get("source", "")),
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
            "acmg_classification": normalize_text(row.get("acmg_classification", "")),
            "acmg_criteria": normalize_text(row.get("acmg_criteria", "")),
            "clinvar_classification": normalize_text(row.get("clinvar_classification", "")),
            "consequence_gene_symbol": normalize_text(row.get("consequence_gene_symbol", "")),
        }

        gene_symbol = normalize_gene_symbol(row.get("consequence_gene_symbol", ""))
        disease_labels = split_multi_delim(row.get("phenotype_combined", ""))
        phenotype_groups = split_multi_delim(row.get("PhenotypeIDS", ""))
        hpo_terms = [normalize_hpo_id(ph_term) for ph_term in phenotype_groups]
        hpo_terms = [ph_term for ph_term in hpo_terms if ph_term]

        valid_disease_terms = []
        for disease_term in disease_labels:
            disease_term = normalize_text(disease_term)
            if not disease_term or disease_term in PLACEHOLDER_VALUES or disease_term.lower() in {
                "not provided",
                "not applicable",
                "not specified",
            }:
                continue
            valid_disease_terms.append(disease_term)
            variant_disease_rels.add((variant_id, disease_term))
            if gene_symbol:
                gene_disease_rels.add((gene_symbol, disease_term))

        for ph_term in hpo_terms:
            variant_phenotype_rels.add((variant_id, ph_term))
            if gene_symbol:
                gene_phenotype_rels.add((gene_symbol, ph_term))
            for disease_term in valid_disease_terms:
                phenotype_disease_rels.add((ph_term, disease_term))

        if gene_symbol:
            gene_variant_rels.add((gene_symbol, variant_id))

    payload = (
        variant_nodes,
        variant_phenotype_rels,
        variant_disease_rels,
        gene_variant_rels,
        gene_disease_rels,
        gene_phenotype_rels,
        phenotype_disease_rels,
    )
    save_pickle_cache(VARIANT_CACHE_PATH, payload)
    LOGGER.info(
        "variant nodes=%s, variant-phenotype=%s, variant-disease=%s, gene-variant=%s",
        len(variant_nodes),
        len(variant_phenotype_rels),
        len(variant_disease_rels),
        len(gene_variant_rels),
    )
    return payload



VARIANT_READ_CHUNKSIZE = 250000
VARIANT_USECOLS = [
    'variant_name',
    'consequence_transcript',
    'consequence_hgvs_c',
    'consequence_hgvs_p',
    'source',
    'Source',
    'chr',
    'pos',
    'ref',
    'alt',
    'consequence_effect',
    'consequence_protein_id',
    'dbsnp',
    'revel_score',
    'revel_prediction',
    'spliceai_max_score',
    'spliceai_max_prediction',
    'acmg_score',
    'acmg_classification',
    'acmg_criteria',
    'clinvar_classification',
    'consequence_gene_symbol',
    'phenotype_combined',
    'PhenotypeIDS',
]


def pick_value(row, idx_map, *keys, default='') -> str:
    for key in keys:
        idx = idx_map.get(key)
        if idx is None:
            continue
        value = row[idx]
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def build_variant_name_from_row(row, idx_map, fallback_index: int) -> str:
    parts = [
        normalize_text(pick_value(row, idx_map, 'consequence_transcript')),
        normalize_text(pick_value(row, idx_map, 'consequence_hgvs_c')),
        normalize_text(pick_value(row, idx_map, 'consequence_hgvs_p')),
    ]
    parts = [p for p in parts if p]
    if parts:
        return ':'.join(parts)

    chr_val = normalize_text(pick_value(row, idx_map, 'chr'))
    pos_val = normalize_text(pick_value(row, idx_map, 'pos'))
    ref_val = normalize_text(pick_value(row, idx_map, 'ref'))
    alt_val = normalize_text(pick_value(row, idx_map, 'alt'))
    if chr_val and pos_val and ref_val and alt_val:
        return f'variant:{chr_val}:{pos_val}:{ref_val}>{alt_val}'
    return f'variant_{fallback_index}'


def load_variant_data():
    LOGGER.info('loading variant data: %s', VARIANT_CSV_FILE_PATH)
    header_df = pd.read_csv(VARIANT_CSV_FILE_PATH, sep='	', dtype=str, keep_default_na=False, encoding='utf-8', nrows=0)
    available_cols = set(header_df.columns)
    usecols = [col for col in VARIANT_USECOLS if col in available_cols]
    chunk_iter = pd.read_csv(
        VARIANT_CSV_FILE_PATH,
        sep='	',
        dtype=str,
        keep_default_na=False,
        encoding='utf-8',
        usecols=usecols,
        chunksize=VARIANT_READ_CHUNKSIZE,
    )

    variant_nodes = OrderedDict()
    variant_phenotype_rels = set()
    variant_disease_rels = set()
    gene_variant_rels = set()
    gene_disease_rels = set()
    gene_phenotype_rels = set()
    phenotype_disease_rels = set()

    total_rows = 0
    for chunk_index, chunk in enumerate(chunk_iter, start=1):
        chunk = chunk.fillna('')
        cols = list(chunk.columns)
        idx_map = {col: i for i, col in enumerate(cols)}
        variant_name_idx = idx_map.get('variant_name')
        source_key = 'source' if 'source' in idx_map else 'Source'
        rows_before = total_rows
        for row_offset, row in enumerate(chunk.itertuples(index=False, name=None), start=1):
            total_rows += 1
            variant_id = ''
            if variant_name_idx is not None:
                variant_id = normalize_text(row[variant_name_idx])
            if not variant_id:
                variant_id = build_variant_name_from_row(row, idx_map, total_rows)
            if not variant_id:
                continue

            variant_nodes[variant_id] = {
                'Source': normalize_text(pick_value(row, idx_map, source_key)),
                'chr': normalize_text(pick_value(row, idx_map, 'chr')),
                'pos': normalize_text(pick_value(row, idx_map, 'pos')),
                'ref': normalize_text(pick_value(row, idx_map, 'ref')),
                'alt': normalize_text(pick_value(row, idx_map, 'alt')),
                'consequence_effect': normalize_text(pick_value(row, idx_map, 'consequence_effect')),
                'consequence_protein_id': normalize_text(pick_value(row, idx_map, 'consequence_protein_id')),
                'dbsnp': normalize_text(pick_value(row, idx_map, 'dbsnp')),
                'revel_score': normalize_text(pick_value(row, idx_map, 'revel_score')),
                'revel_prediction': normalize_text(pick_value(row, idx_map, 'revel_prediction')),
                'spliceai_max_score': normalize_text(pick_value(row, idx_map, 'spliceai_max_score')),
                'spliceai_max_prediction': normalize_text(pick_value(row, idx_map, 'spliceai_max_prediction')),
                'acmg_score': normalize_text(pick_value(row, idx_map, 'acmg_score')),
                'acmg_classification': normalize_text(pick_value(row, idx_map, 'acmg_classification')),
                'acmg_criteria': normalize_text(pick_value(row, idx_map, 'acmg_criteria')),
                'clinvar_classification': normalize_text(pick_value(row, idx_map, 'clinvar_classification')),
                'consequence_gene_symbol': normalize_text(pick_value(row, idx_map, 'consequence_gene_symbol')),
            }

            gene_symbol = normalize_gene_symbol(pick_value(row, idx_map, 'consequence_gene_symbol'))
            disease_labels = split_pipe(pick_value(row, idx_map, 'phenotype_combined'))
            phenotype_groups = split_pipe_comma(pick_value(row, idx_map, 'PhenotypeIDS'))
            hpo_terms = [normalize_hpo_id(ph_term) for ph_term in phenotype_groups]
            hpo_terms = [ph_term for ph_term in hpo_terms if ph_term]

            valid_disease_terms = []
            for disease_term in disease_labels:
                disease_term = normalize_text(disease_term)
                if not disease_term or disease_term in PLACEHOLDER_VALUES or disease_term.lower() in {
                    'not provided',
                    'not applicable',
                    'not specified',
                }:
                    continue
                valid_disease_terms.append(disease_term)
                variant_disease_rels.add((variant_id, disease_term))
                if gene_symbol:
                    gene_disease_rels.add((gene_symbol, disease_term))

            for ph_term in hpo_terms:
                variant_phenotype_rels.add((variant_id, ph_term))
                if gene_symbol:
                    gene_phenotype_rels.add((gene_symbol, ph_term))
                for disease_term in valid_disease_terms:
                    phenotype_disease_rels.add((ph_term, disease_term))

            if gene_symbol:
                gene_variant_rels.add((gene_symbol, variant_id))

        LOGGER.info('processed chunk %s rows=%s total_rows=%s variants=%s', chunk_index, len(chunk), total_rows, len(variant_nodes))

    payload = (
        variant_nodes,
        variant_phenotype_rels,
        variant_disease_rels,
        gene_variant_rels,
        gene_disease_rels,
        gene_phenotype_rels,
        phenotype_disease_rels,
    )
    save_pickle_cache(VARIANT_CACHE_PATH, payload)
    LOGGER.info(
        'variant nodes=%s, variant-phenotype=%s, variant-disease=%s, gene-variant=%s',
        len(variant_nodes),
        len(variant_phenotype_rels),
        len(variant_disease_rels),
        len(gene_variant_rels),
    )
    return payload


def main() -> None:
    (
        variant_nodes,
        variant_phenotype_rels,
        variant_disease_rels,
        gene_variant_rels,
        gene_disease_rels,
        gene_phenotype_rels,
        phenotype_disease_rels,
    ) = load_variant_data()

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
        for vid, props in variant_nodes.items()
    ]
    variant_rows = dedupe_preserve_order(variant_rows, ["variant_name:ID"])

    vp_rows = [
        {":START_ID": vid, ":END_ID": ph, "TYPE": "IS_ASSOCIATED_WITH"}
        for vid, ph in sorted(variant_phenotype_rels)
    ]
    vd_rows = [
        {":START_ID": vid, ":END_ID": disease_term, "TYPE": "ASSOCIATED_WITH_DISEASE"}
        for vid, disease_term in sorted(variant_disease_rels)
    ]
    export_csv(OUT_DIR / "variants.csv", [
        "variant_name:ID", "Source", "chr", "pos", "ref", "alt", "consequence_effect",
        "consequence_protein_id", "dbsnp", "revel_score", "revel_prediction",
        "spliceai_max_score", "spliceai_max_prediction", "acmg_score", "acmg_criteria",
        "clinvar_classification", "consequence_gene_symbol"
    ], variant_rows)
    export_csv(OUT_DIR / "variant_phenotype.csv", [":START_ID", ":END_ID", "TYPE"], vp_rows)
    export_csv(OUT_DIR / "variant_disease.csv", [":START_ID", ":END_ID", "TYPE"], vd_rows)
    export_csv(OUT_DIR / "gene_variant.csv", [":START_ID", ":END_ID", "TYPE"], [
        {":START_ID": gs, ":END_ID": vid, "TYPE": "HAS_VARIANT"} for gs, vid in sorted(gene_variant_rels)
    ])
    export_csv(OUT_DIR / "gene_disease.csv", [":START_ID", ":END_ID", "TYPE"], [
        {":START_ID": gs, ":END_ID": disease, "TYPE": "ASSOCIATED_WITH_DISEASE"} for gs, disease in sorted(gene_disease_rels)
    ])
    export_csv(OUT_DIR / "gene_phenotype.csv", [":START_ID", ":END_ID", "TYPE"], [
        {":START_ID": gs, ":END_ID": ph, "TYPE": "ASSOCIATED_WITH_PHENOTYPE"} for gs, ph in sorted(gene_phenotype_rels)
    ])
    export_csv(OUT_DIR / "phenotype_disease.csv", [":START_ID", ":END_ID", "TYPE"], [
        {":START_ID": ph, ":END_ID": disease, "TYPE": "ASSOCIATED_WITH_DISEASE"} for ph, disease in sorted(phenotype_disease_rels)
    ])

    LOGGER.info("clinvar complete")


if __name__ == "__main__":
    main()

