# RareSeek-R1
### Domain-Specialized Large Language Model for Rare-Disease Diagnostic Reasoning

[![Model Weights](https://img.shields.io/badge/Model_Weights-Huggingface-FFD21E.svg)](https://huggingface.co/TaoMedAI/RareSeek-R1)
[![Dataset](https://img.shields.io/badge/Dataset-RareMedData-blue.svg)](https://huggingface.co/datasets/TaoMedAI/RareMedData)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-Non_Commercial-red.svg)](#license)

**RareSeek-R1** is a domain-specialized language model for rare-disease diagnostic reasoning, developed through Progressive Parameter-Efficient Transfer Learning. It is instruction-tuned on **RareMed-Corpus**, integrating textbooks, guidelines, biomedical literature and EHR narratives, then fine-tuned on **RareMed-CoT** for stepwise clinical reasoning.

![Overview of RareSeek-R1](https://github.com/TaoMedAI/RareSeek-R1/blob/main/RareSeek-R1.png)
*Figure 1. Framework and pipeline of RareSeek-R1.*

[Data](#data) · [Installation](#installation) · [Usage](#usage) · [Code](#code) · [License](#license)

## Data

| Resource | Download |
| :--- | :--- |
| Model weights | [TaoMedAI/RareSeek-R1](https://huggingface.co/TaoMedAI/RareSeek-R1) — request access and save to `models/RareSeek-R1/` |
| Training data & KG components | [TaoMedAI/RareMedData](https://huggingface.co/datasets/TaoMedAI/RareMedData) |
| ClinVar · GRCh38, 2025.02.09 | [NCBI FTP](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/) |
| HGMD · 2024.02 | [HGMD](https://www.hgmd.cf.ac.uk/ac/index.php) — licensed access required |

Place graph sources in `RareMed-KG/data/raw/` and clinical inputs in `data/private/`. Patient records and HGMD content are not included. Use matching data and model versions when reproducing study results.

## Installation

Linux and NVIDIA GPUs are required for vLLM inference. The study uses four H800 GPUs with PyTorch 2.6.0, CUDA 12.6 and vLLM 0.8.4.

```bash
git clone https://github.com/TaoMedAI/RareSeek-R1.git
cd RareSeek-R1
conda create -n rareseek python=3.10
conda activate rareseek
pip install -r requirements.txt
```

## Usage

**EHR → Phenotypes → Inference → Disease identifiers → Evaluation**

Run commands from the repository root. Input files use UTF-8 JSONL: one record per line, with a unique `sample_id` and clinical narrative in `context`. RAG additionally reads positive HPO IDs from `hpo_list` and candidate variants from `variants`.

### 1. EHR preprocessing

For non-English records, set their `language` field and run the optional translator. English records (`language=en`) are passed through.

```bash
pip install -r requirements/services.txt
export DEEPSEEK_API_KEY="YOUR_KEY"
python rare_disease_ehr_translator.py --input data/private/cases_raw.jsonl --output data/private/cases_en.jsonl
```

Translation and HPO services send text to their configured backend; use authorized records or a suitable local service.

### 2. Phenotype extraction

Download [BioLORD-2023-M](https://huggingface.co/FremyCompany/BioLORD-2023-M) to `BioLORD/`. Start the extraction service, then run its client in another terminal:

```bash
uvicorn llm_hpo_extraction_service:app --host 127.0.0.1 --port 8000
python llm_hpo_api_client.py --input data/private/cases_en.jsonl --output data/private/cases.jsonl
```

The default backend is [DeepSeek](https://platform.deepseek.com/). Set `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` and `DEEPSEEK_API_KEY` for a compatible endpoint, including [OpenAI](https://platform.openai.com/) or a local [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) deployment. Review positive HPO mappings before inference.

<details>
<summary>Phenotype extraction baselines</summary>

| Tool | Study parameters |
| :--- | :--- |
| [PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger) | `model_type=pubmedbert`, `onlyLongest=True`, `abbrRecog=True`, `negation=True`, `ML_Threshold=0.95` |
| [PhenoBERT](https://github.com/EclipseCN/PhenoBERT) | `param1=0.8`, `param2=0.6`, `param3=0.9`, `use_longest=True`, `use_step_3=True` |

Follow each tool's installation instructions and store positive HPO IDs in `hpo_list`.

</details>

### 3. Standard inference

Supply clinical records and local model weights directly; no configuration file or reference labels are required.

```bash
python inference.py --input data/private/cases.jsonl --model models/RareSeek-R1 --output outputs/standard
```

Predictions are saved to `outputs/standard/predictions.jsonl`. For four GPUs, add `--tensor-parallel-size 4`. Slurm users can pass the same arguments to `sbatch inference/inference.sh`.

### 4. Knowledge graph setup

<details>
<summary>Source files and variant preparation</summary>

Under `RareMed-KG/data/raw/`, provide:

- `Orphapacket/*.json`: disease records under an `Orphapacket` object.
- `HPO_OMIM/genes_to_disease.json`: gene-keyed disease associations.
- `HPO_OMIM/HPO_phenotype_KG.json`: disease-keyed records with `hpo_list` entries containing `id` and `hpo_term`.
- `HPO_OMIM/phenotype_to_genes.json`: HPO-keyed records with `hpo_name` and `genes` associations.
- `phenotype_to_genes_20210413.txt`: seven tab-separated columns for HPO ID, label, Entrez ID, gene symbol, additional information, source and disease ID.
- `clinvar_20250209.vcf.gz` and the corresponding GRCh38 `variant_summary.txt`.
- `HGMD_20242.geneanno.tsv` when using licensed HGMD data.

Set `GENEBE_USERNAME` and `GENEBE_API_KEY` for variant annotation:

```bash
pip install -r requirements/annotation.txt
python RareMed-KG/integration_clinvar_hgmd/acquire_clinvar.py
python RareMed-KG/integration_clinvar_hgmd/clinvar_annotate.py
```

For HGMD, additionally run `format_hgmd.py`, `map_hgmd_coords.py` and `process_hgmd_annotation.py` from the same directory. Then combine sources:

```bash
python RareMed-KG/integration_clinvar_hgmd/reconcile_databases.py --sources clinvar-hgmd
python RareMed-KG/integration_clinvar_hgmd/unify_variants.py
```

Use `--sources clinvar` for ClinVar only. The integration output is `RareMed-KG/data/processed/integration/variants_HGMD_Clinvar.tsv`. Intermediate annotation files are tab-separated, including those named `.csv`.

</details>

Build the graph, export its snapshot and import it into an **empty** Neo4j database:

```bash
python scripts/build_kg.py
python scripts/graph_snapshot.py --directory RareMed-KG/data/processed/neo4j_import --output data/private --data-kind private
export NEO4J_PASSWORD="YOUR_DATABASE_PASSWORD"
docker compose up -d
python scripts/import_neo4j.py
```

This produces `graph.json`, `hierarchy.json` and `HPO_phenotype.json` in `data/private/`. Neo4j defaults to `bolt://localhost:7687`; set `NEO4J_URI`, `NEO4J_USERNAME` and `NEO4J_DATABASE` as needed. Each graph snapshot requires a fresh database.

### 5. RareMedRAG

```bash
python RareMedRAG.py --input data/private/cases.jsonl --output outputs/rag/raw_predictions.jsonl --model models/RareSeek-R1 --revision YOUR_WEIGHT_VERSION --ic-source data/private/HPO_phenotype.json --tensor-parallel-size 4
```

RAG queries all candidate variants and their unique genes, then retrieves phenotype paths. The model receives at most **300 paths** in strict priority order: **variants → genes → phenotypes**. Phenotype paths fill the remaining slots in descending mean IC order, using **IC(h) = −ln(nₕ/N)** without smoothing.

Variant entries use `Gene.refGene`, `AAChange.refGene` (or `hgvs.c`), `Chr`, `Start`, `End`, `Ref` and `Alt`. The IC snapshot must match the imported graph. `--path-limit`, `--max-model-len` and `--max-tokens` control retrieval and context limits; each prediction records its selected evidence.

### 6. Post-processing & evaluation

```bash
python rare_disease_entity_extractor.py --input outputs/rag/raw_predictions.jsonl --output outputs/rag/extracted.jsonl
python scripts/normalize_diseases.py --input outputs/rag/extracted.jsonl --output outputs/rag/predictions.jsonl --graph data/private/graph.json
python scripts/evaluate.py --gold data/private/gold.jsonl --predictions outputs/rag/predictions.jsonl --hierarchy data/private/hierarchy.json --output outputs/rag/metrics.json
```

Reference records contain `sample_id` and a `true_label` list of ORPHA/OMIM identifiers. Local normalization uses graph names and aliases; `monarch_api_client.py --query "DISEASE_NAME"` offers an optional Monarch lookup for review.

Exact and hierarchical evaluation use the full reference cohort, including missing or empty predictions. Hierarchical matching includes direct parents, children and grandchildren for ORPHA labels. `evaluation.ipynb` uses the same evaluator. To use `Classifications of rare diseases.json`, convert it with `scripts/convert_hierarchy.py --output data/private/hierarchy.json`.

## Code

| Location | Purpose |
| :--- | :--- |
| `inference.py` / `RareMedRAG.py` | Standard inference / graph-augmented inference |
| `RareMed-KG/` | Source integration and graph construction |
| `rareseek/` | Shared file handling, IC, retrieval formatting and evaluation |
| `scripts/` | Graph import, disease normalization and evaluation commands |
| `scripts/run_three_hop.py` | Combined construction, import, RAG and evaluation; use `--help` for arguments |

## License

For **non-commercial research and academic use only**. Third-party resources retain their own licenses. For model access, datasets or academic collaborations, contact the corresponding author listed in the paper or open an issue in this repository.
