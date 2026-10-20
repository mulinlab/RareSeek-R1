from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rareseek.legacy_paths import RAW, PROCESSED, INTEGRATION, integration_dir


import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = RAW / 'HPO_OMIM'
OUT_DIR = PROCESSED / 'HPO_OMIM'

GENE_DISEASE_JSON = BASE_DIR / 'genes_to_disease.json'
HPO_PHENOTYPE_JSON = BASE_DIR / 'HPO_phenotype_KG.json'
PHENOTYPE_TO_GENES_JSON = BASE_DIR / 'phenotype_to_genes.json'


def _join_list(value: Any) -> str:
    if isinstance(value, list):
        return ', '.join(str(x) for x in value)
    if value is None:
        return ''
    return str(value)


def _save_csv(rows: list[dict], output_path: Path) -> None:
    schemas = {
        'diseases_extracted.csv': ['disease_id','disease_name','aliases'],
        'gene_disease_kg_extracted.csv': ['gene','association_type','disease_id','disease_name'],
        'hpo_phenotype_kg_extracted.csv': ['disease_id','disease_name','hpo_id','hpo_term','chpo_term'],
        'phenotype_to_genes_kg_extracted.csv': ['gene','hpo_id','hpo_name','frequency','disease_id','disease_name']}
    df = pd.DataFrame(rows, columns=schemas[output_path.name])
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'saved {output_path} rows={len(df)}')


def extract_gene_disease() -> None:
    with GENE_DISEASE_JSON.open('r', encoding='utf-8') as f:
        data = json.load(f)

    rows_list = []
    for gene, associations in data.items():
        for association in associations:
            row = {
                'gene': gene,
                'association_type': association.get('association_type'),
                'disease_id': association.get('disease_id'),
                'disease_name': _join_list(association.get('disease_name', [])),
            }
            rows_list.append(row)

    _save_csv(rows_list, OUT_DIR / 'gene_disease_kg_extracted.csv')


def extract_hpo_phenotype() -> None:
    with HPO_PHENOTYPE_JSON.open('r', encoding='utf-8') as f:
        data = json.load(f)

    rows_list = []
    for disease_id, disease_info in data.items():
        disease_name_str = _join_list(disease_info.get('disease_name', ['Unknown']))
        for hpo_item in disease_info.get('hpo_list', []) or []:
            if hpo_item.get('excluded') is True or str(hpo_item.get('qualifier', '')).strip().upper() == 'NOT':
                continue
            rows_list.append({
                'disease_id': disease_id,
                'disease_name': disease_name_str,
                'hpo_id': hpo_item.get('id'),
                'hpo_term': hpo_item.get('hpo_term'),
                'chpo_term': hpo_item.get('chpo_term', ''),
            })

    _save_csv(rows_list, OUT_DIR / 'hpo_phenotype_kg_extracted.csv')


def extract_phenotype_to_genes() -> None:
    with PHENOTYPE_TO_GENES_JSON.open('r', encoding='utf-8') as f:
        data = json.load(f)

    rows_list = []
    seen = set()
    for hpo_id, hpo_info in data.items():
        hpo_name = hpo_info.get('hpo_name', 'Unknown')
        for gene_info in hpo_info.get('genes', []):
            gene_symbol = gene_info.get('gene_symbol') or ''
            disease_id = gene_info.get('disease_id') or ''
            disease_name = _join_list(gene_info.get('disease_name', [])) if isinstance(gene_info.get('disease_name'), list) else gene_info.get('disease_name', '')
            row = {
                'gene': gene_symbol,
                'hpo_id': hpo_id,
                'hpo_name': hpo_name,
                'frequency': '-',
                'disease_id': disease_id,
                'disease_name': disease_name,
            }
            key = (row['gene'], row['hpo_id'], row['disease_id'], row['disease_name'])
            if key in seen:
                continue
            seen.add(key)
            rows_list.append(row)

    _save_csv(rows_list, OUT_DIR / 'phenotype_to_genes_kg_extracted.csv')


def extract_disease_catalog():
    catalog = {}
    def add(identifier, names):
        if not isinstance(identifier, str) or not identifier.strip():
            return
        names = names if isinstance(names, list) else [names]
        catalog.setdefault(identifier, set()).update(n.strip() for n in names if isinstance(n, str) and n.strip() and n != 'Unknown')
    with GENE_DISEASE_JSON.open(encoding='utf-8') as stream:
        for associations in json.load(stream).values():
            for association in associations:
                add(association.get('disease_id'), association.get('disease_name', []))
    with HPO_PHENOTYPE_JSON.open(encoding='utf-8') as stream:
        for identifier, record in json.load(stream).items():
            add(identifier, record.get('disease_name', []))
    with PHENOTYPE_TO_GENES_JSON.open(encoding='utf-8') as stream:
        for record in json.load(stream).values():
            for association in record.get('genes', []):
                add(association.get('disease_id'), association.get('disease_name', []))
    rows = []
    for identifier, names in sorted(catalog.items()):
        names = sorted(names)
        rows.append({'disease_id': identifier, 'disease_name': names[0] if names else identifier,
                     'aliases': '|'.join(names[1:])})
    _save_csv(rows, OUT_DIR / 'diseases_extracted.csv')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extract_disease_catalog()
    extract_gene_disease()
    extract_hpo_phenotype()
    extract_phenotype_to_genes()


if __name__ == '__main__':
    main()
