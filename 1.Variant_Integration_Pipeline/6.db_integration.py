import pandas as pd
import re

columns_keep = [
    "chr","pos","ref","alt","effect","transcript","gene_symbol","gene_hgnc_id",
    "consequences_refseq","dbsnp","revel_score","revel_prediction",
    "spliceai_max_score","spliceai_max_prediction","acmg_score",
    "acmg_classification","acmg_criteria","clinvar_classification",
    "phenotype_combined"
]

chr_list = list(range(1,23)) + ["X","Y","MT"]

def extract(pattern, text):
    m = re.search(pattern, str(text))
    return m.group(1) if m else "."

for chr in chr_list:

    ### ----------- read ClinVar data ----------- 
    clinvar = pd.read_csv(f"clinvar_20250209_review.Chr{chr}.csv", sep="\t")
    clinvar = clinvar[columns_keep]
    clinvar["source"] = "clinvar"

    ### ----------- read HGMD data ----------- 
    if chr != "MT":
        hgmd = pd.read_csv(f"HGMD_20242_Genebe_annotate_result.Chr{chr}.csv", sep="\t")
        hgmd = hgmd[columns_keep]
        hgmd["source"] = "HGMD"
        df = pd.concat([clinvar, hgmd])
    else:
        df = clinvar

    ### ----------- remove duplicate variants ----------- 
    df = df.drop_duplicates(subset=["chr","pos","ref","alt"])
    df = df[df["chr"] == chr].reset_index(drop=True)

    ### ----------- split the consequence field ----------- 
    df["consequences_refseq"] = (
        df["consequences_refseq"]
        .str.replace("}, ", "},, ")
        .str.split(",, ")
    )

    df = df.explode("consequences_refseq").reset_index(drop=True)

    rows = []

    for _, r in df.iterrows():

        c = r["consequences_refseq"]

        effect = extract(r"consequences': \['([A-Za-z0-9_,\' ]+)'\]", c).replace("', '","|")
        gene = extract(r"'gene_symbol': '([A-Za-z0-9_,\' -.]+)', '", c)
        hgnc = extract(r"gene_hgnc_id': ([0-9]+),", c)
        transcript = extract(r"transcript': '([A-Za-z0-9_.]+)',", c)
        protein = extract(r"protein_id': '([A-Za-z0-9_.]+)',", c)
        hgvs_c = extract(r"hgvs_c': '([A-Za-z0-9_.>+*-]+)',", c)
        hgvs_p = extract(r"hgvs_p': '([A-Za-z0-9_.>+*-?]+)',", c)

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

    final_df.to_csv(
        f"./variant/chr{chr}.HGMD_and_clinvar.csv",
        sep="\t",
        index=False
    )