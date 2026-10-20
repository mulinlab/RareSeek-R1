# RareMed-KG

Knowledge graph preparation, variant integration and Neo4j import for RareMedRAG.

All commands below run from the **RareSeek-R1 repository root**, with the environment described in the main README activated.

## Data directories

```text
RareMed-KG/
├── data/
│   ├── raw/                    # Source data
│   │   ├── Orphapacket/
│   │   └── HPO_OMIM/
│   └── processed/              # Intermediate files and graph exports
│       ├── integration/
│       └── neo4j_import/
├── integration_clinvar_hgmd/
└── KG_construction/
```

Place these source files under `RareMed-KG/data/raw/`:

| Path | Contents |
| :--- | :--- |
| `Orphapacket/*.json` | Disease records under an `Orphapacket` object |
| `HPO_OMIM/genes_to_disease.json` | Gene-keyed disease associations |
| `HPO_OMIM/HPO_phenotype_KG.json` | Disease-keyed records; `hpo_list` entries contain `id` and `hpo_term` |
| `HPO_OMIM/phenotype_to_genes.json` | HPO-keyed records with `hpo_name` and `genes` associations |
| `phenotype_to_genes_20210413.txt` | Seven-column TSV: HPO ID, label, Entrez ID, gene symbol, additional information, source, disease ID |
| `clinvar_20250209.vcf.gz` | ClinVar GRCh38 VCF |
| `variant_summary.txt` | Corresponding ClinVar GRCh38 variant summary |
| `HGMD_20242.geneanno.tsv` | Licensed HGMD input, when available |

Download ClinVar from [NCBI](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/), general graph components from [RareMedData](https://huggingface.co/datasets/TaoMedAI/RareMedData), and obtain HGMD through [HGMD](https://www.hgmd.cf.ac.uk/ac/index.php). HGMD requires the appropriate license. Source data and generated graph files are excluded from version control.

## 1. Prepare ClinVar

Install the annotation dependencies and set `GENEBE_USERNAME` and `GENEBE_API_KEY` in your local environment.

```bash
pip install -r requirements/annotation.txt
python RareMed-KG/integration_clinvar_hgmd/acquire_clinvar.py
python RareMed-KG/integration_clinvar_hgmd/clinvar_annotate.py
```

The first script extracts chromosome-level variant files; the second annotates them. Outputs are written under `data/processed/integration/` within `RareMed-KG`.

## 2. Prepare HGMD (optional)

When using licensed HGMD data, run these scripts in order:

```bash
python RareMed-KG/integration_clinvar_hgmd/format_hgmd.py
python RareMed-KG/integration_clinvar_hgmd/map_hgmd_coords.py
python RareMed-KG/integration_clinvar_hgmd/process_hgmd_annotation.py
```

They prepare HGVS records, map them to genomic coordinates and annotate the resulting variants. Skip this section for ClinVar-only construction.

## 3. Integrate variants

For ClinVar and HGMD together:

```bash
python RareMed-KG/integration_clinvar_hgmd/reconcile_databases.py --sources clinvar-hgmd
python RareMed-KG/integration_clinvar_hgmd/unify_variants.py
```

For ClinVar only, use the following commands instead:

```bash
python RareMed-KG/integration_clinvar_hgmd/reconcile_databases.py --sources clinvar
python RareMed-KG/integration_clinvar_hgmd/unify_variants.py
```

Both modes produce `RareMed-KG/data/processed/integration/variants_HGMD_Clinvar.tsv`. This filename is shared by both modes; it does not imply that HGMD was used. Intermediate annotation files are tab-separated, including files with a `.csv` suffix.

## 4. Build the graph

```bash
python scripts/build_kg.py
```

This entry point runs the following scripts in order and validates the graph exports:

| Order | Script in `KG_construction/` | Purpose |
| :--- | :--- | :--- |
| 1 | `extract_orphanet.py` | Prepare Orphanet entities and associations |
| 2 | `extract_hpo_omim.py` | Prepare HPO, OMIM and gene associations |
| 3 | `extract_variant.py` | Prepare integrated variant entities and associations |
| 4 | `load_neo4j.py` | Assemble and export graph CSV files |

Do not rerun these four scripts individually after running the entry point. Despite its name, `load_neo4j.py` exports CSV files; database import is a separate step.

The final CSV files and build manifest are saved to `RareMed-KG/data/processed/neo4j_import/`. Other intermediate subdirectories are created automatically.

## 5. Export the snapshot and import Neo4j

```bash
python scripts/graph_snapshot.py --directory RareMed-KG/data/processed/neo4j_import --output data/private --data-kind private
export NEO4J_PASSWORD="YOUR_DATABASE_PASSWORD"
docker compose up -d
python scripts/import_neo4j.py
```

Import into an **empty Neo4j database**. The default URI is `bolt://localhost:7687`; configure `NEO4J_URI`, `NEO4J_USERNAME` and `NEO4J_DATABASE` when needed.

Snapshot files `graph.json`, `hierarchy.json` and `HPO_phenotype.json` are written to the repository-level `data/private/` directory. Use the `HPO_phenotype.json` from this same graph build as the IC source for RareMedRAG. See the main README for inference commands.
