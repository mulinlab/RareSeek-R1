# RareSeek-R1: Domain-Specialized Large Language Model for Rare-Disease Diagnostic Reasoning

[![Model Weights](https://img.shields.io/badge/Model_Weights-Huggingface-FFD21E.svg)](https://huggingface.co/TaoMedAI/RareSeek-R1)
[![Dataset](https://img.shields.io/badge/Dataset-RareMedData-blue.svg)](https://huggingface.co/datasets/TaoMedAI/RareMedData)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-Non_Commercial-red.svg)](#license)

**RareSeek-R1** is a domain-specialized large language model for rare-disease diagnostic reasoning, developed through a Progressive Parameter-Efficient Transfer Learning framework.

The model is first instruction-tuned on **RareMed-Corpus**, which integrates medical textbooks, guidelines, biomedical literature, and EHR narratives. It is subsequently fine-tuned on **RareMed-CoT** to support explicit, stepwise clinical reasoning.

![Overview of RareSeek-R1](https://github.com/TaoMedAI/RareSeek-R1/blob/main/RareSeek-R1.png)
*Figure 1. Overall framework and pipeline of RareSeek-R1.*

---

## 📑 Table of Contents

- [Data Availability](#data-availability)
- [Getting Started](#getting-started)
- [Data Organization](#data-organization)
- [Usage Pipeline](#usage-pipeline)
  - [1. EHR Preprocessing](#1-ehr-preprocessing)
  - [2. Phenotype Extraction](#2-phenotype-extraction)
  - [3. Knowledge Graph Construction](#3-knowledge-graph-construction)
  - [4. Standard Model Inference](#4-standard-model-inference)
  - [5. RareMedRAG](#5-raremedrag)
  - [6. Post-processing and Evaluation](#6-post-processing-and-evaluation)
- [Code Reference](#code-reference)
- [License and Contact](#license)

<a id="data-availability"></a>
## 📊 Data Availability

| Resource | Access | Use |
| :--- | :--- | :--- |
| **RareMedData** | [TaoMedAI/RareMedData](https://huggingface.co/datasets/TaoMedAI/RareMedData) | Training data and available knowledge graph components |
| **RareSeek-R1 weights** | [TaoMedAI/RareSeek-R1](https://huggingface.co/TaoMedAI/RareSeek-R1) | Model inference; request access through the model page |
| **ClinVar** | [NCBI GRCh38 VCF directory](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/) | Genetic variation data; scripts use the 2025.02.09 release naming convention |
| **HGMD** | [Human Gene Mutation Database](https://www.hgmd.cf.ac.uk/ac/index.php) | Licensed variation data; scripts use the 2024.02 release naming convention |

Patient-level EHR records and licensed HGMD content are not distributed with this repository. Supply authorized copies locally. A ClinVar-only graph and a graph incorporating HGMD are different experimental settings; record the sources and versions used for each run. Obtain the corresponding GRCh38 `variant_summary.txt` alongside the ClinVar VCF.

<a id="getting-started"></a>
## 🚀 Getting Started

### Environment Setup

Model inference targets **Linux with NVIDIA GPUs**. The study configuration uses four NVIDIA H800 GPUs; adjust tensor parallelism and context length to the model and available GPU memory. The dependency files specify PyTorch 2.6.0 and vLLM 0.8.4; use a compatible NVIDIA driver and CUDA runtime (the study environment used CUDA 12.6).

```bash
git clone https://github.com/TaoMedAI/RareSeek-R1.git
cd RareSeek-R1
conda create -n rareseek python=3.10
conda activate rareseek
python -m pip install -r requirements.txt
```

Optional dependencies are separated by task:

| File | Purpose |
| :--- | :--- |
| `requirements/kg.txt` | CPU graph construction and Neo4j driver |
| `requirements/inference.txt` | Local model inference, including graph dependencies |
| `requirements/services.txt` | Translation and HPO extraction services |
| `requirements/annotation.txt` | GeneBe variant annotation |

### Model Weights

Request access at [HuggingFace: TaoMedAI/RareSeek-R1](https://huggingface.co/TaoMedAI/RareSeek-R1), then download the weights to `models/RareSeek-R1/`. Record the model revision or a checksum identifier for the local weights. For access questions, contact the corresponding author listed in the paper.

<a id="data-organization"></a>
## 🗂 Data Organization

Create the following directories as needed; source data, clinical records and weights are supplied separately.

```text
RareSeek-R1/
├── models/RareSeek-R1/                 # Local model weights
├── BioLORD/                            # Phenotype embedding model
├── data/private/                       # Clinical inputs, reference labels, graph snapshots
├── RareMed-KG/data/
│   ├── raw/                            # Downloaded and licensed source files
│   └── processed/
│       ├── integration/                # Standardized variant annotations
│       └── neo4j_import/               # Node and relationship CSVs
├── configs/                            # Run configurations
└── outputs/                            # Predictions, metrics and run manifests
```

**Paths:** command-line data paths are relative to the repository root unless absolute. Paths inside JSON configurations are relative to the configuration file. Graph preparation also accepts `RARESEEK_RAW_DIR` and `RARESEEK_PROCESSED_DIR`; use these consistently across integration and construction commands.

**Case input:** UTF-8 JSONL, one record per line. No reference labels should appear in model inputs.

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `sample_id` | Nonempty string | Unique case identifier |
| `context` | Nonempty string | Clinical narrative |
| `language` | String | `en` for English; required by the translation command |
| `hpo_list` | List of strings | Confirmed positive HPO IDs; use an empty list when none are available |
| `variants` | List of objects | Candidate variants; use an empty list when absent |

RareMedRAG reads `Gene.refGene`, `AAChange.refGene` (or `hgvs.c`), `Chr`, `Start`, `End`, `Ref`, and `Alt` from each variant. HGVS entries must include a supported transcript identifier and a `c.` change; transcript versions are preserved. Multiple genes may be separated by commas or semicolons. Set `context` to the English narrative before running RareMedRAG.

Reference labels belong in a separate `gold.jsonl`: each record contains `sample_id` and a nonempty `true_label` list of disease identifiers (`ORPHA:` and/or `OMIM:`). Predictions use `predict_code`, a ranked list of lists: one inner list per rank; an unmapped rank is an empty list.

<a id="usage-pipeline"></a>
## 🛠 Usage Pipeline

**Clinical inputs → preprocessing → HPO extraction → inference → disease normalization → evaluation**

RareMedRAG additionally requires a constructed and imported knowledge graph. Run the commands below from the repository root.

### 1. EHR Preprocessing

The translation command processes records explicitly marked as non-English and leaves `language=en` records unchanged.

```bash
python -m pip install -r requirements/services.txt
export DEEPSEEK_API_KEY="YOUR_KEY"
python rare_disease_ehr_translator.py --input data/private/cases_raw.jsonl --output data/private/cases_en.jsonl
```

This optional command sends non-English narrative text to its configured DeepSeek service. Use only records authorized for that destination, or prepare translations locally before inference.

### 2. Phenotype Extraction

#### A. Traditional Baselines

The study includes [PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger) and [PhenoBERT](https://github.com/EclipseCN/PhenoBERT). Install and run these tools following their repositories, then place confirmed positive HPO IDs in `hpo_list`.

| Algorithm | Parameter | Value |
| :--- | :--- | :--- |
| PhenoTagger | `model_type` | `pubmedbert` |
| | `onlyLongest` | `True` |
| | `abbrRecog` | `True` |
| | `negation` | `True` |
| | `ML_Threshold` | `0.95` |
| PhenoBERT | `param1` | `0.8` |
| | `param2` | `0.6` |
| | `param3` | `0.9` |
| | `use_longest` | `True` |
| | `use_step_3` | `True` |

#### B. LLM-based HPO Extraction

Download [BioLORD-2023-M](https://huggingface.co/FremyCompany/BioLORD-2023-M) to `BioLORD/`. The service maps extracted phrases to the terminology in `zh-HPO.json`.

A locally deployed [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B), [OpenAI](https://platform.openai.com/) or [DeepSeek APIs](https://platform.deepseek.com/) can provide a compatible backend. The service defaults to DeepSeek; configure its endpoint and model explicitly for another provider.

```bash
export DEEPSEEK_API_KEY="YOUR_KEY"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"
export BIOLORD_PATH="$(pwd)/BioLORD"
uvicorn llm_hpo_extraction_service:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
python llm_hpo_api_client.py --input data/private/cases_en.jsonl --output data/private/cases.jsonl --url http://localhost:8000
```

The client stores the extraction details and positive mapped IDs. Review the extraction, negation status and mappings before using them for diagnostic evaluation.

### 3. Knowledge Graph Construction

#### Source files

Place the following under `RareMed-KG/data/raw/`:

| Source | Required structure |
| :--- | :--- |
| `Orphapacket/*.json` | An `Orphapacket` object containing disease identifiers and available genes, phenotypes and parents |
| `HPO_OMIM/genes_to_disease.json` | Gene-keyed associations with `disease_id`, `disease_name`, and optional `association_type` |
| `HPO_OMIM/HPO_phenotype_KG.json` | Disease-keyed records with `disease_name` and `hpo_list` objects containing `id`, `hpo_term`, optional `chpo_term` |
| `HPO_OMIM/phenotype_to_genes.json` | HPO-keyed records with `hpo_name` and `genes` objects containing `gene_symbol`, `disease_id`, and optional `disease_name` |
| `phenotype_to_genes_20210413.txt` | Seven-column tab-separated table: HPO ID, HPO label, Entrez gene ID, gene symbol, additional information, source, disease ID |
| `clinvar_20250209.vcf.gz` | GRCh38 ClinVar source VCF |
| `variant_summary.txt` | ClinVar summary with `Assembly`, chromosome/VCF coordinates and `PhenotypeIDS` |
| `HGMD_20242.geneanno.tsv` | Licensed HGMD input, only when using the HGMD branch |

Explicitly excluded HPO entries (`excluded=true` or `qualifier=NOT`) are not positive disease annotations. Disease–phenotype frequency is stored on the relationship. Co-occurrence within a variant record is not promoted to a direct disease–phenotype annotation.

#### Variant integration

GeneBe annotation requires the annotation dependencies and credentials:

```bash
python -m pip install -r requirements/annotation.txt
export GENEBE_USERNAME="YOUR_USERNAME"
export GENEBE_API_KEY="YOUR_KEY"
python RareMed-KG/integration_clinvar_hgmd/acquire_clinvar.py
python RareMed-KG/integration_clinvar_hgmd/clinvar_annotate.py
```

For a licensed HGMD source, also run:

```bash
python RareMed-KG/integration_clinvar_hgmd/format_hgmd.py
python RareMed-KG/integration_clinvar_hgmd/map_hgmd_coords.py
python RareMed-KG/integration_clinvar_hgmd/process_hgmd_annotation.py
python RareMed-KG/integration_clinvar_hgmd/reconcile_databases.py --sources clinvar-hgmd
```

For ClinVar only, use `--sources clinvar` instead. `--chromosomes` selects the chromosomes present in the prepared annotation files. The `.csv` intermediate annotation files are tab-separated; do not change their delimiter based on their extension.

```bash
python RareMed-KG/integration_clinvar_hgmd/unify_variants.py
python scripts/build_kg.py --raw RareMed-KG/data/raw --processed RareMed-KG/data/processed
python scripts/import_neo4j.py --validate-only
python scripts/graph_snapshot.py --directory RareMed-KG/data/processed/neo4j_import --output data/private --data-kind private
```

The integration output is `RareMed-KG/data/processed/integration/variants_HGMD_Clinvar.tsv`. Its `source` column records provenance, including when only ClinVar is used. Conflicting attributes for the same variant identifier require source reconciliation before building the graph.

The snapshot command writes:

- `graph.json`: disease records and aliases for local identifier normalization.
- `hierarchy.json`: disease identifiers mapped to their direct parents.
- `HPO_phenotype.json`: disease–HPO annotations used for IC calculation.

#### Neo4j import

```bash
export NEO4J_PASSWORD="YOUR_DATABASE_PASSWORD"
docker compose up -d
python scripts/import_neo4j.py --database neo4j
```

The container uses Neo4j 5.26.0 Community and binds its ports to localhost. Use a dedicated **empty database** for each graph snapshot. The importer verifies node IDs, relationship types and endpoints before writing. It does not clear an existing database; a partial import also requires a fresh database before retrying.

Set `NEO4J_URI`, `NEO4J_USERNAME` and `NEO4J_DATABASE` when using another server; the importer also accepts `--uri`, `--username`, and `--database`. No APOC extension is required.

### 4. Standard Model Inference

Configure the input, reference labels, normalization graph, hierarchy and model version:

```bash
cp configs/private.example.json configs/study.local.json
# Edit configs/study.local.json with the paths and model revision for this run.
python run_pipeline.py preflight --config configs/study.local.json
python inference.py --config configs/study.local.json
```

The configuration uses `mode=standard` and `backend=vllm`. Standard inference does not query Neo4j; the local graph file is used to normalize predicted disease names for evaluation. English input may omit a translation file. For non-English input, supply a local translation mapping through the optional `translations` configuration path.

Results include `predictions.jsonl`, `metrics.json`, `prompts.jsonl`, and `manifest.json` in the configured output directory. Use a new output directory for each experiment.

**Slurm batch inference:**

```bash
sbatch inference/inference.sh "$(pwd)/configs/study.local.json"
```

**Multiple configured runs:**

```bash
python scripts/run_experiments.py --configs configs/study.local.json configs/comparison.local.json --output outputs/summary.csv
```

Each configuration must use a distinct output directory.

### 5. RareMedRAG

RareMedRAG retrieves paths from the configured Neo4j graph and places them alongside the clinical narrative in the model prompt.

```bash
python RareMedRAG.py \
  --input data/private/cases.jsonl \
  --output outputs/rag/raw_predictions.jsonl \
  --model "$(pwd)/models/RareSeek-R1" \
  --revision YOUR_WEIGHT_REVISION \
  --ic-source data/private/HPO_phenotype.json \
  --database neo4j \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --max-tokens 8192
```

**Retrieval and context allocation**

- Every input variant is queried. Gene identifiers from all variants are deduplicated and queried separately.
- Each query traverses undirected paths of one to three graph relationships. This is graph-neighborhood retrieval; the construction workflow does not import HPO `IS_A` hierarchy edges.
- The prompt includes at most **300 path records**, in strict order: **variant paths → gene paths → phenotype paths**. Variant paths consume the budget first, followed by genes. Phenotypes use the remaining budget in descending IC score order.
- Variant paths follow the input-variant order and a deterministic path order per target; genes use sorted identifiers. No quota is reserved for later categories. If variant paths fill the budget, gene and phenotype paths are omitted.
- A phenotype path is scored by the mean IC of its distinct phenotype nodes, including zero values. `IC(h) = -ln(n_h / N)`, where `N` is the number of disease records, including those without annotations, and `n_h` counts records annotated with that HPO term. There is no smoothing or automatic ancestor propagation.
- Path IC measures annotation specificity, not a patient's diagnostic probability. A phenotype occurring on a retrieved path is not necessarily present in the patient. Evidence records preserve nodes, relationship types and directions.

The IC file must match the imported disease population and direct disease–phenotype annotations. Phenotypes without IC coverage cause an error rather than receiving an invented score. Each query has a `--path-limit` guard (default 2,000); increase it explicitly if the complete neighborhood exceeds that limit. The token budget is checked separately from the 300-path limit.

Each prediction stores the retrieval context and a `retrieval_report` with available, selected, omitted and unmatched-target counts. Unmatched targets do not occupy path slots.

**Combined construction-to-evaluation command**

After preparing the raw source files, integrated variant TSV and an empty Neo4j database, the following command runs graph construction, snapshot creation, import, RAG inference, normalization and evaluation:

```bash
python scripts/run_three_hop.py \
  --raw RareMed-KG/data/raw \
  --processed RareMed-KG/data/processed \
  --cases data/private/cases.jsonl \
  --gold data/private/gold.jsonl \
  --output outputs/study-run-001 \
  --model "$(pwd)/models/RareSeek-R1" \
  --revision YOUR_WEIGHT_REVISION \
  --tensor-parallel-size 4 \
  --database neo4j
```

Use this command **instead of** importing the graph manually first. Reproducing study metrics requires the corresponding cohort, source snapshots, weights, preprocessing and evaluation protocol; record their versions and checksums. Model and database execution must be validated in the intended server environment.

### 6. Post-processing and Evaluation

For a standalone RareMedRAG run:

```bash
python rare_disease_entity_extractor.py --input outputs/rag/raw_predictions.jsonl --output outputs/rag/extracted.jsonl
python scripts/normalize_diseases.py --input outputs/rag/extracted.jsonl --output outputs/rag/predictions.jsonl --graph data/private/graph.json
python scripts/evaluate.py --gold data/private/gold.jsonl --predictions outputs/rag/predictions.jsonl --hierarchy data/private/hierarchy.json --output outputs/rag/metrics.json
```

The entity parser reads the final numbered disease list. Local normalization matches disease names and aliases against the fixed graph snapshot; unknown or ambiguous names remain unmapped and keep their ranking positions.

`monarch_api_client.py --query "DISEASE_NAME"` provides an optional Monarch lookup. Its returned identifiers are candidates for review; it is not used automatically for benchmark scoring.

Both **Exact Evaluation** and **Hierarchical Evaluation** use the complete reference cohort. Missing and empty predictions count as misses. Duplicate or foreign sample IDs and malformed prediction schemas are rejected. Hierarchical matching includes exact labels and, for ORPHA labels, direct parents, children and grandchildren.

To use the supplied Orphanet classification tree:

```bash
python scripts/convert_hierarchy.py --input "Classifications of rare diseases.json" --output data/private/hierarchy.json
```

Choose the hierarchy before evaluation and keep it consistent across runs. `evaluation.ipynb` calls the same evaluator; edit its paths to point to your local inputs and predictions.

<a id="code-reference"></a>
## 🧭 Code Reference

| Location | Role |
| :--- | :--- |
| `RareMedRAG.py` | Neo4j retrieval, IC ranking, prompt assembly and local RAG inference |
| `inference.py`, `inference/` | Standard inference and Slurm/model-serving entry points |
| `rareseek/io.py` | UTF-8 file access, consistent paths, ID validation and checksums |
| `rareseek/ic.py`, `rareseek/rag_support.py` | IC computation, path allocation and evidence formatting |
| `rareseek/pipeline.py`, `rareseek/graph.py` | Standard inference and local disease identifier mapping |
| `rareseek/evaluation.py` | Exact and hierarchical evaluation |
| `RareMed-KG/integration_clinvar_hgmd/` | ClinVar/HGMD preparation and variant annotation |
| `RareMed-KG/KG_construction/` | Disease, phenotype, gene and variant graph construction |
| `scripts/build_kg.py`, `scripts/import_neo4j.py`, `scripts/graph_snapshot.py` | Graph orchestration, import and snapshot export |
| `scripts/run_three_hop.py` | Combined graph construction, RAG and evaluation workflow |
| `rare_disease_ehr_translator.py`, `llm_hpo_*` | Optional translation and HPO extraction services |
| `Exomiser_Config.yml`, `scripts/render_exomiser.py` | Exomiser configuration template and local rendering |

<a id="license"></a>
## 📜 License & Contact

- **License:** Code and model weights are provided for non-commercial research and academic use only. Third-party datasets, tools and model components remain subject to their respective access terms and licenses.
- **Contact:** For academic collaborations, dataset inquiries or model access, refer to the corresponding author listed in the paper or open an issue in this repository.
