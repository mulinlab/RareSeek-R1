import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rareseek.legacy_paths import RAW, PROCESSED, INTEGRATION, integration_dir

import pandas as pd
import genebe as gnb
from rareseek.variants import annotation_frame

# ======== Configuration ========
import os
API_USERNAME = os.environ.get("GENEBE_USERNAME", "")
API_KEY = os.environ.get("GENEBE_API_KEY", "")
ENDPOINT = "https://api.genebe.net/cloud/api-public/v1/variants"
GENOME = "hg38"

chr_list = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]

# ======== Main Process ========
def main():
    if not API_USERNAME or not API_KEY:
        raise ValueError("Set GENEBE_USERNAME and GENEBE_API_KEY; no network call made")
    if not list(INTEGRATION.glob("HGMD_20242_Genebe_HGVStoPOS.Chr*.csv")):
        raise FileNotFoundError("No mapped HGMD files; run map_hgmd_coords.py first")
    for chr in chr_list:
        print(f"Processing chromosome {chr} ...")
    
        if not (INTEGRATION / f"HGMD_20242_Genebe_HGVStoPOS.Chr{chr}.csv").exists():
            continue
        # Read and split coordinates
        df = pd.read_csv(INTEGRATION / f"HGMD_20242_Genebe_HGVStoPOS.Chr{chr}.csv", sep="\t", dtype=str, keep_default_na=False)
        if not {"chr", "pos", "ref", "alt"}.issubset(df.columns):
            if len(df.columns) != 1:
                raise ValueError("Expected chr/pos/ref/alt columns or one hyphen-delimited coordinate column")
            df = df.iloc[:, 0].str.split("-", n=3, expand=True)
            df.columns = ["chr", "pos", "ref", "alt"]
        df = df[["chr", "pos", "ref", "alt"]]
        if df.empty:
            df.assign(consequences_refseq='').to_csv(
                integration_dir() / f"HGMD_20242_Genebe_annotate_result.Chr{chr}.csv", sep='\t', index=False)
            continue
    
        # Force chr column to string to avoid type issues
        df['chr'] = df['chr'].astype(str)
    
        try:
            # Perform batch annotation
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
            # Save successful results
            annotated.to_csv(
                integration_dir() / f"HGMD_20242_Genebe_annotate_result.Chr{chr}.csv",
                index=False,
                sep="\t"
            )
    
        except Exception as e:
            print(f"Annotation failed for entire chromosome {chr}: {e}")
            # Save original input as failure record
            df.to_csv(
                integration_dir() / f"HGMD_20242_Genebe_annotate_result.Chr{chr}.fail_to_annotate.csv",
                index=False,
                sep="\t"
            )
            raise

if __name__ == "__main__":
    main()
