#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Purpose: Process ClinVar VCF file, keep only Pathogenic / Likely pathogenic variants
         Split by chromosome and generate TSV files with standard header
"""

import pandas as pd
import os
from pathlib import Path
import gzip
import re
from typing import List

# ==================== Configuration ====================
INPUT_VCF_GZ     = "clinvar_20250209.vcf.gz"
OUTPUT_PREFIX    = "clinvar_20250209.vcf.pathogenic_or_likely_pathogenic"
CHROMS           = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]

WANTED_CLNSIG_KEYWORDS = ["pathogenic", "likely_pathogenic", "likely pathogenic"]
UNWANTED_KEYWORD       = "conflicting"

# Final desired header (based on the order in your original script)
FINAL_HEADER = [
    "chrom", "pos", "ref", "alt", "qual", "filter",
    "af_esp", "af_exac", "af_tgp",
    "alleleid", "clndids", "clndn", "clnhgvs", "clnrevstat",
    "clnsig", "clnsigscv", "clnvc", "clnvcso", "clnvi",
    "geneinfo", "mc", "origin", "rs",
    "supplement1", "supplement2", "supplement3", "sup4",
    "sup5", "sup6", "sup7", "sup8"
]

# ==================== Helper Functions ====================
def parse_info_field(info_str: str) -> dict:
    """Parse INFO field, keep KEY=VALUE format in the value itself"""
    info_dict = {}
    for field in info_str.split(";"):
        if "=" in field:
            k, v = field.split("=", 1)
            info_dict[k] = f"{k}={v}"
        else:
            info_dict[field] = True
    return info_dict


def read_clinvar_vcf_gz(file_path: str):
    """Read vcf.gz, keep only #CHROM POS ID REF ALT QUAL FILTER INFO columns"""
    data = []
    with gzip.open(file_path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                # VCF standard header line
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos, rsid, ref, alt, qual, filt, info = fields[:8]
            row = {
                "#CHROM": chrom,
                "POS": pos,
                "ID": rsid,
                "REF": ref,
                "ALT": alt,
                "QUAL": qual,
                "FILTER": filt,
                "INFO": info
            }
            data.append(row)
    return pd.DataFrame(data)


def extract_info_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expand INFO field into multiple columns"""
    info_series = df["INFO"].apply(parse_info_field)
    info_df = pd.json_normalize(info_series)

    # Merge back to main table and drop original INFO column
    df = pd.concat([df.drop(columns=["INFO"]), info_df], axis=1)

    # Convert column names to lowercase (for easier filtering)
    df.columns = [c.lower() for c in df.columns]

    # Rename commonly used columns to match desired header
    rename_map = {
        "#chrom": "chrom",
        "pos":    "pos",
        "id":     "rs",
        "ref":    "ref",
        "alt":    "alt",
        "qual":   "qual",
        "filter": "filter",
        "af_esp": "af_esp",
        "af_exac": "af_exac",
        "af_tgp":  "af_tgp",
        "alleleid": "alleleid",   # note: ClinVar often uses 'alleleid'
        "clndn":   "clndn",
        "clnhgvs": "clnhgvs",
        "clnrevstat": "clnrevstat",
        "clnsig":  "clnsig",
        "clnvc":   "clnvc",
        "clnvcso": "clnvcso",
        "clnvi":   "clnvi",
        "geneinfo": "geneinfo",
        "mc":      "mc",
        "origin":  "origin",
    }
    df = df.rename(columns=rename_map)

    return df


def is_pathogenic(row) -> bool:
    unwanted = UNWANTED_KEYWORD.lower()

    def contains_any(text: str, keywords: list) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(k.lower() in t for k in keywords)

    def contains_unwanted(text: str) -> bool:
        return contains_any(text, [unwanted])

    def contains_wanted(text: str) -> bool:
        return contains_any(text, WANTED_CLNSIG_KEYWORDS)

    # CLNSIG
    clnsig = row.get("clnsig", "")
    if pd.isna(clnsig):
        clnsig = ""
    if contains_unwanted(clnsig):
        return False
    if contains_wanted(clnsig):
        return True

    # CLNSIGINCL
    clnsigincl = row.get("clnsigincl", "")
    if pd.isna(clnsigincl):
        clnsigincl = ""
    else:
        clnsigincl = str(clnsigincl).strip()

    if not clnsigincl:
        return False

    for part in clnsigincl.split("|"):
        part = part.strip()
        if ":" not in part:
            continue
        _, sig = part.split(":", 1)
        sig = sig.strip()
        if contains_unwanted(sig):
            return False
        if contains_wanted(sig):
            return True

    return False


# ==================== Main Workflow ====================
def main():
    print("Reading ClinVar file...")
    df = read_clinvar_vcf_gz(INPUT_VCF_GZ)

    print("Expanding INFO field...")
    df = extract_info_columns(df)

    print("Filtering for Pathogenic / Likely pathogenic variants (including CLNSIGINCL)...")
    mask = df.apply(is_pathogenic, axis=1)
    df_path = df[mask].copy()

    print(f"After filtering, {len(df_path):,} records remain")

    # Ensure chrom is string type and sort for consistent output
    df_path["chrom"] = df_path["chrom"].astype(str)
    df_path = df_path.sort_values(["chrom", "pos"])

    # Split by chromosome and save
    out_dir = Path("clinvar_split")
    out_dir.mkdir(exist_ok=True)

    for chrom in CHROMS:
        df_chr = df_path[df_path["chrom"] == chrom].copy()

        if df_chr.empty:
            print(f"chr{chrom}: No qualifying variants, skipping")
            continue

        # Fill missing columns with empty string
        for col in FINAL_HEADER:
            if col not in df_chr.columns:
                df_chr[col] = ""

        # Reorder columns according to FINAL_HEADER
        df_chr = df_chr[FINAL_HEADER]

        outfile = out_dir / f"{OUTPUT_PREFIX}.chr{chrom}.tsv"
        df_chr.to_csv(outfile, sep="\t", index=False, na_rep="")
        print(f"Saved: {outfile} ({len(df_chr):,} rows)")


if __name__ == "__main__":
    main()