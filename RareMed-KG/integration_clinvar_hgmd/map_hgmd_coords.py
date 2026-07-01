import pandas as pd
import genebe as gnb


API_USERNAME = "your_username"
API_KEY = "your_api_key"

CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y"]

for chrom in CHROMOSOMES:
    
    # 1. Read HGVS list for this chromosome
    file_path = f"HGMD_20242.geneAnno.chr{chrom}_for_Genebe.tsv"
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
    output_file = f"HGMD_20242_Genebe_HGVStoPOS.Chr{chrom}.csv"
    pd.DataFrame(parsed).to_csv(output_file, sep="\t", index=False)
    print(f"Saved: {output_file}  ({len(parsed)} records)")