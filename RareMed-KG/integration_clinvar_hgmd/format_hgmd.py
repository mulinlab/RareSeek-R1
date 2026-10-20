import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rareseek.legacy_paths import RAW, PROCESSED, INTEGRATION, integration_dir

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
from rareseek.variants import chromosome
from pathlib import Path


INPUT_FILE = RAW / "HGMD_20242.geneanno.tsv"
OUTPUT_PREFIX = "HGMD_20242.geneAnno"

# List of chromosomes to process
CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]


def main():
    print("Reading input file...")

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        header=0,
        usecols=range(8),
        dtype=str,
        low_memory=False
    )

    df.iloc[:, 0] = df.iloc[:, 0].map(chromosome)
    print("Start processing by chromosome...")

    for chrom in CHROMS:
        print(f"  Processing chr{chrom} ...", end=" ", flush=True)

        # 1. Filter rows for this chromosome
        df_chr = df[df.iloc[:, 0] == chrom].copy()

        if df_chr.empty:
            output = integration_dir() / f"{OUTPUT_PREFIX}.chr{chrom}_for_Genebe.tsv"
            output.write_text('', encoding='utf-8')
            print('(no variants)')
            continue

        # 2. Sort by position (column 1, natural string sort)
        df_chr = df_chr.sort_values(by=df_chr.columns[1], key=lambda x: x.astype(str))

        # Reorder columns and insert "HGMD" constant column
        df_out = df_chr.iloc[:, [0, 1, 2, 3, 4, 5, 7, 6]].copy()
        df_out.insert(1, "HGMD", "HGMD")

        # 3. Prepare format for Genebe: take columns 5 and 6 of df_out → pos:c.change
        df_genebe = df_out.iloc[:, [5, 6]].copy()
        changes = df_genebe.iloc[:, 1].fillna('').astype(str).str.strip()
        transcripts = df_genebe.iloc[:, 0].fillna('').astype(str).str.strip()
        if (changes == '').any() or (transcripts == '').any():
            raise ValueError('HGMD transcript and cDNA columns must be populated')
        changes = changes.map(lambda v: v if re.match(r'^[cngm]\.', v) else 'c.' + v)
        df_genebe['variant'] = transcripts + ':' + changes

        output = integration_dir() / f"{OUTPUT_PREFIX}.chr{chrom}_for_Genebe.tsv"
        df_genebe["variant"].to_csv(output, sep="\t", index=False, header=False)
        print(f" {output} ({len(df_genebe):,} rows)")


if __name__ == "__main__":
    main()