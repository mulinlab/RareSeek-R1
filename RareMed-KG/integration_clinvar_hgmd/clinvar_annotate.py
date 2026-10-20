import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rareseek.legacy_paths import RAW, PROCESSED, INTEGRATION, integration_dir

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import genebe as gnb
from rareseek.variants import annotation_frame
from pathlib import Path
from tqdm import tqdm


# ================= Configuration =================
import os
API_USERNAME = os.environ.get("GENEBE_USERNAME", "")
API_KEY = os.environ.get("GENEBE_API_KEY", "")

ENDPOINT = "https://api.genebe.net/cloud/api-public/v1/variants"
GENOME   = "hg38"

INPUT_PATTERN  = "clinvar_20250209.vcf.pathogenic_or_likely_pathogenic.chr{chr}.tsv"
OUTPUT_PATTERN = "clinvar_20250209_review.Chr{chr}.csv"

CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
# ==================================================


def get_expected_columns():
    """Return the expected column names of the input file"""
    return ["chr", "pos", "ref", "alt"]


def process_one_chromosome(chr_name: str) -> None:
    input_path = INTEGRATION / "clinvar_split" / INPUT_PATTERN.format(chr=chr_name)
    output_path = integration_dir() / OUTPUT_PATTERN.format(chr=chr_name)

    if not input_path.exists():
        print(f"Skipping {chr_name}: file does not exist → {input_path}")
        return

    try:
        df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)[["chrom", "pos", "ref", "alt"]].rename(columns={"chrom": "chr"})
        df["pos"] = df["pos"].astype(int)
        if df.empty:
            df.assign(consequences_refseq='').to_csv(output_path, sep='\t', index=False)
            print(f'Chr{chr_name}: no variants to annotate')
            return

        print(f"Chr{chr_name} annotation started... ({len(df):,} variants)")

        annotated = gnb.annotate(
            variants=df,
            flatten_consequences=False,
            username=API_USERNAME,
            api_key=API_KEY,
            endpoint_url=ENDPOINT,
            output_format="dataframe",
            genome=GENOME,
            use_ensembl=False,
            use_refseq=True
        )

        annotated = annotation_frame(annotated, df)
        annotated.to_csv(output_path, sep="\t", index=False)
        print(f"→ Saved: {output_path} ({len(annotated):,} rows)")

    except Exception as e:
        print(f"Chr{chr_name} processing failed: {e}")
        raise


def main():
    if not API_USERNAME or not API_KEY:
        raise ValueError("Set GENEBE_USERNAME and GENEBE_API_KEY; no network call made")
    if not any((INTEGRATION / "clinvar_split" / INPUT_PATTERN.format(chr=c)).exists() for c in CHROMOSOMES):
        raise FileNotFoundError("No ClinVar split files; run acquire_clinvar.py first")
    print(f"Starting processing of {len(CHROMOSOMES)} chromosomes")
    print(f"Using account: {API_USERNAME}\n")

    for chrom in tqdm(CHROMOSOMES, desc="Chromosome progress"):
        process_one_chromosome(chrom)

    print("\nAll chromosomes processed")


if __name__ == "__main__":
    main()