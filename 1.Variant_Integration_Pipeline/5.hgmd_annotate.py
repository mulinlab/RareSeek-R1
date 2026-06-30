import pandas as pd
import genebe as gnb

# ======== Configuration ========
API_USERNAME = "your_username"
API_KEY = "your_api_key"
ENDPOINT = "https://api.genebe.net/cloud/api-public/v1/variants"
GENOME = "hg38"

chr_list = [str(i) for i in range(1, 23)] + ["X", "Y"]

# ======== Main Process ========
for chr in chr_list:
    print(f"Processing chromosome {chr} ...")

    # Read and split coordinates
    df = pd.read_csv(f"HGMD_20242_Genebe_HGVStoPOS.Chr{chr}.csv", header=None)
    df = df[0].str.split("-", expand=True)
    df.columns = ['chr', 'pos', 'ref', 'alt']

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

        # Save successful results
        annotated.to_csv(
            f"HGMD_20242_Genebe_annotate_result.Chr{chr}.csv",
            index=False,
            sep="\t"
        )

    except Exception as e:
        print(f"Annotation failed for entire chromosome {chr}: {e}")
        # Save original input as failure record
        df.to_csv(
            f"HGMD_20242_Genebe_annotate_result.Chr{chr}.fail_to_annotate.csv",
            index=False,
            sep="\t"
        )