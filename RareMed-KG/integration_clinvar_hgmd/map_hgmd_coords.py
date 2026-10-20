import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rareseek.legacy_paths import RAW, PROCESSED, INTEGRATION, integration_dir

import pandas as pd
import genebe as gnb
from rareseek.variants import coordinates


import os
API_USERNAME = os.environ.get("GENEBE_USERNAME", "")
API_KEY = os.environ.get("GENEBE_API_KEY", "")

CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]

def main():
    if not API_USERNAME or not API_KEY:
        raise ValueError("Set GENEBE_USERNAME and GENEBE_API_KEY; no network call made")
    if not list(INTEGRATION.glob("HGMD_20242.geneAnno.chr*_for_Genebe.tsv")):
        raise FileNotFoundError("No formatted HGMD files; run format_hgmd.py first")
    for chrom in CHROMOSOMES:
        
        # 1. Read HGVS list for this chromosome
        file_path = INTEGRATION / f"HGMD_20242.geneAnno.chr{chrom}_for_Genebe.tsv"
        if not file_path.exists():
            continue
        if file_path.stat().st_size == 0:
            pd.DataFrame(columns=['chr', 'pos', 'ref', 'alt']).to_csv(
                integration_dir() / f"HGMD_20242_Genebe_HGVStoPOS.Chr{chrom}.csv", sep='\t', index=False)
            continue
        df = pd.read_csv(file_path, sep="\t", header=None, names=["NM_cDNA"])
        hgvs_list = df["NM_cDNA"].tolist()
    
        if not hgvs_list:
            print(f"chr{chrom}: file is empty, skipping")
            continue
    
        # 2. Call Genebe API to parse HGVS → genomic coordinates
        print(f"Parsing chr{chrom} ... {len(hgvs_list)} variants")
        parsed = gnb.parse_variants(
            variants=hgvs_list,
            genome="hg38",
            username=API_USERNAME,
            api_key=API_KEY
        )
    
        # 3. Save result as tsv (avoid calling API repeatedly)
        output_file = integration_dir() / f"HGMD_20242_Genebe_HGVStoPOS.Chr{chrom}.csv"
        if len(parsed) != len(hgvs_list):
            raise ValueError(f'chr{chrom}: coordinate conversion returned an incomplete result')
        records = []
        for value in parsed:
            if isinstance(value, str):
                parts = value.split('-', 3)
                if len(parts) != 4:
                    raise ValueError('Coordinate conversion returned an unresolved variant')
                value = dict(zip(['chr', 'pos', 'ref', 'alt'], parts))
            if not isinstance(value, dict):
                raise ValueError('Coordinate conversion must return coordinates for every input')
            normalized = coordinates(value)
            if normalized is None:
                raise ValueError('Coordinate conversion returned incomplete coordinates')
            records.append(normalized)
        pd.DataFrame(records, columns=['chr', 'pos', 'ref', 'alt']).to_csv(output_file, sep="\t", index=False)
        print(f"Saved: {output_file}  ({len(parsed)} records)")

if __name__ == "__main__":
    main()
