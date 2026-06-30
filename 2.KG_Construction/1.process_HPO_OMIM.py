from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path('/mnt/hpc/home/scllm/deepseek_model/rare_disease/new/complete/data/json_KG')
OUT_DIR = Path('/mnt/hpc/home/scllm/deepseek_model/rare_disease/new/complete/json_KG')
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'saved {output_path} rows={len(df)}')


def extract_gene_disease() -> None:
    with GENE_DISEASE_JSON.open('r', encoding='utf-8') as f:
        data = json.load(f)

    rows_list = []
    for gene, associations in data.items():
        for association in associations:
            row = {
                'disease': gene,
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
        for hpo_item in disease_info.get('hpo_list', []):
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


def probe_file_type(file_path: str) -> None:
    try:
        import magic  # type: ignore
    except Exception as exc:
        print(f'magic unavailable: {exc}')
        return

    file_type = magic.from_file(file_path)
    print(f'file_type: {file_type}')
    if 'pickle' in file_type.lower():
        import pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(type(data), len(data) if hasattr(data, '__len__') else 'n/a')
    elif 'json' in file_type.lower():
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(type(data), len(data) if hasattr(data, '__len__') else 'n/a')
    elif 'text' in file_type.lower():
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f.read(500))
    else:
        with open(file_path, 'rb') as f:
            print(f.read(100))


def main() -> None:
    extract_gene_disease()
    extract_hpo_phenotype()
    extract_phenotype_to_genes()


if __name__ == '__main__':
    main()
