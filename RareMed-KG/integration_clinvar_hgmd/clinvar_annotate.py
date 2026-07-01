#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import genebe as gnb
from pathlib import Path
from tqdm import tqdm


# ================= Configuration =================
API_USERNAME = "your_username"
API_KEY     = "your_api_key"

ENDPOINT = "https://api.genebe.net/cloud/api-public/v1/variants"
GENOME   = "hg38"

INPUT_PATTERN  = "clinvar_20250209.vcf.pathgenic_or_likely_pathogenic.Chr{chr}.txt"
OUTPUT_PATTERN = "clinvar_20250209_review.Chr{chr}.csv"

CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
# ==================================================


def get_expected_columns():
    """Return the expected column names of the input file"""
    return ["chr", "pos", "ref", "alt"]


def process_one_chromosome(chr_name: str) -> None:
    input_path = Path(INPUT_PATTERN.format(chr=chr_name))
    output_path = Path(OUTPUT_PATTERN.format(chr=chr_name))

    if not input_path.exists():
        print(f"Skipping {chr_name}: file does not exist → {input_path}")
        return

    try:
        df = pd.read_csv(
            input_path,
            sep="\t",
            header=None,
            names=get_expected_columns(),
            dtype={"chr": str, "pos": int, "ref": str, "alt": str},
            low_memory=False
        )

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

        annotated.to_csv(output_path, sep="\t", index=False)
        print(f"→ Saved: {output_path} ({len(annotated):,} rows)")

    except Exception as e:
        print(f"Chr{chr_name} processing failed: {e}")


def main():
    print(f"Starting processing of {len(CHROMOSOMES)} chromosomes")
    print(f"Using account: {API_USERNAME}\n")

    for chrom in tqdm(CHROMOSOMES, desc="Chromosome progress"):
        process_one_chromosome(chrom)

    print("\nAll chromosomes processed")


if __name__ == "__main__":
    main()