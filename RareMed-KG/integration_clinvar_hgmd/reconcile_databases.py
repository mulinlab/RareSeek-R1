import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rareseek.legacy_paths import RAW, PROCESSED, INTEGRATION, integration_dir

import pandas as pd
import re
import json
import ast
from rareseek.variants import chromosome
from rareseek.io import write_json

columns_keep = [
    "chr","pos","ref","alt","effect","transcript","gene_symbol","gene_hgnc_id","hgvs_c","hgvs_p",
    "consequences_refseq","dbsnp","revel_score","revel_prediction",
    "spliceai_max_score","spliceai_max_prediction","acmg_score",
    "acmg_classification","acmg_criteria","clinvar_classification",
    "phenotype_combined"
]

chr_list = [str(i) for i in range(1,23)] + ["X","Y","MT"]

def parse_consequences(value):
    if not value or value in {'.', 'null', 'None'}:
        return [{}]
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or any(not isinstance(x, dict) for x in parsed):
        raise ValueError('consequences_refseq must contain a list of objects')
    return parsed or [{}]

def field(record, key):
    value = record.get(key)
    if value is None or value == '':
        return '.'
    return '|'.join(map(str, value)) if isinstance(value, list) else str(value)

def read_annotations(filename):
    frame = pd.read_csv(filename, sep='\t', dtype=str, keep_default_na=False)
    if not {'chr', 'pos', 'ref', 'alt'}.issubset(frame.columns):
        raise ValueError(f'{filename}: annotation coordinates missing')
    if 'consequences_refseq' not in frame:
        frame['consequences_refseq'] = ''
    for column in columns_keep:
        if column not in frame:
            frame[column] = ''
    frame['chr'] = frame['chr'].map(chromosome)
    return frame


def main():
    import argparse
    p = argparse.ArgumentParser(description="Reconcile explicitly selected sources; HGMD is never supplied by this repository")
    p.add_argument("--sources", choices=["clinvar", "clinvar-hgmd"], required=True)
    p.add_argument("--chromosomes", nargs="+", default=chr_list)
    args = p.parse_args()
    (integration_dir() / "variant").mkdir(exist_ok=True)
    output_files = []
    write_json(INTEGRATION / 'variant/chunks.json', {'status': 'running', 'files': []})
    for chr in args.chromosomes:
        chr = chromosome(chr)
    
        ### ----------- read ClinVar data ----------- 
        clinvar = read_annotations(INTEGRATION / f"clinvar_20250209_review.Chr{chr}.csv")
        clinvar = clinvar[columns_keep]
        clinvar["source"] = "clinvar"
    
        ### ----------- read HGMD data ----------- 
        if args.sources == "clinvar-hgmd":
            hgmd = read_annotations(INTEGRATION / f"HGMD_20242_Genebe_annotate_result.Chr{chr}.csv")
            hgmd = hgmd[columns_keep]
            hgmd["source"] = "HGMD"
            df = pd.concat([clinvar, hgmd])
        else:
            df = clinvar
    
        ### ----------- remove duplicate variants ----------- 
        df = df.drop_duplicates()
        df = df[df["chr"] == chr].reset_index(drop=True)
    
        ### ----------- split the consequence field ----------- 
        df["consequences_refseq"] = df["consequences_refseq"].map(parse_consequences)

        df = df.explode("consequences_refseq").reset_index(drop=True)
    
        rows = []
    
        for _, r in df.iterrows():
    
            c = r["consequences_refseq"]
    
            effect = field(c, 'consequences') if c.get('consequences') is not None else (r['effect'] or '.')
            gene = field(c, 'gene_symbol') if c.get('gene_symbol') is not None else (r['gene_symbol'] or '.')
            hgnc = field(c, 'gene_hgnc_id') if c.get('gene_hgnc_id') is not None else (r['gene_hgnc_id'] or '.')
            transcript = field(c, 'transcript') if c.get('transcript') is not None else (r['transcript'] or '.')
            protein = field(c, 'protein_id')
            hgvs_c = field(c, 'hgvs_c') if c.get('hgvs_c') is not None else (r['hgvs_c'] or '.')
            hgvs_p = field(c, 'hgvs_p') if c.get('hgvs_p') is not None else (r['hgvs_p'] or '.')

            rows.append([
                r["source"], r["chr"], r["pos"], r["ref"], r["alt"],
                effect, gene, hgnc, transcript, protein, hgvs_c, hgvs_p,
                r["dbsnp"] or ".", r["revel_score"] or ".", r["revel_prediction"] or ".",
                r["spliceai_max_score"] or ".", r["spliceai_max_prediction"] or ".",
                r["acmg_score"] or ".", r["acmg_classification"] or ".",
                r["acmg_criteria"] or ".", r["clinvar_classification"] or ".",
                r["phenotype_combined"] or "."
            ])
    
        final_df = pd.DataFrame(rows, columns=[
            "source","chr","pos","ref","alt","consequence_effect",
            "consequence_gene_symbol","consequence_hgnc_id",
            "consequence_transcript","consequence_protein_id",
            "consequence_hgvs_c","consequence_hgvs_p","dbsnp",
            "revel_score","revel_prediction","spliceai_max_score",
            "spliceai_max_prediction","acmg_score","acmg_classification",
            "acmg_criteria","clinvar_classification","phenotype_combined"
        ])
    
        filename = f"chr{chr}.HGMD_and_clinvar.csv"
        final_df.to_csv(
            INTEGRATION / "variant" / filename,
            sep="\t",
            index=False
        )
        output_files.append(filename)
    write_json(INTEGRATION / 'variant/chunks.json', {'status': 'complete', 'sources': args.sources, 'files': output_files})

if __name__ == "__main__":
    main()
