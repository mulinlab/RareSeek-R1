# RareSeek-R1: Domain-Specialized Large Language Model for Rare-Disease Diagnostic Reasoning

[![Model Weights](https://img.shields.io/badge/Model_Weights-Huggingface-FFD21E.svg)](https://huggingface.co/TaoMedAI/RareSeek-R1)
[![Dataset](https://img.shields.io/badge/Dataset-RareMedData-blue.svg)](https://huggingface.co/datasets/TaoMedAI/RareMedData)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-Non_Commercial-red.svg)](#license)

**RareSeek-R1** is a domain-specialized large language model tailored for rare-disease diagnostic reasoning, developed through a Progressive Parameter-Efficient Transfer Learning framework. 

The model is first instruction-tuned on the clinically grounded **RareMed-Corpus**, a large, multi-source dataset deeply integrated from medical textbooks, guidelines, biomedical literature, and real-world EHR (Electronic Health Record) narratives. It is subsequently fine-tuned on **RareMed-CoT**, a high-fidelity corpus designed to instill explicit, stepwise clinical reasoning that aligns with real-world diagnostic workflows.

![Overview of RareSeek-R1]([https://github.com/TaoMedAI/RareSeek-R1](https://github.com/mulinlab/RareSeek-R1)/blob/main/RareSeek-R1.png)
*Figure 1: Overall framework and pipeline of RareSeek-R1.*

---

## 📑 Table of Contents

- [Data Availability](#data-availability)
- [Getting Started](#getting-started)
  - [Environment Setup](#environment-setup)
  - [Model Weights](#model-weights)
- [Usage Pipeline](#usage-pipeline)
  - [1. EHR Preprocessing](#1-ehr-preprocessing)
  - [2. Phenotype Extraction](#2-phenotype-extraction-baselines--llms)
  - [3. Model Inference (Standard)](#3-model-inference-standard)
  - [4. RareMedRAG](#4-raremedrag)
  - [5. Post-processing & Normalization](#5-post-processing--normalization)
  - [6. Evaluation](#6-evaluation)
- [Code Reference](#code-reference)
- [License & Contact](#license)

---

<a id="data-availability"></a>
## 📊 Data Availability

The open-source data underpinning this study, including training datasets and supplementary knowledge graph components, are publicly available.

- **RareMedData (Training Data & General KG)**: Available at [HuggingFace: TaoMedAI/RareMedData](https://huggingface.co/datasets/TaoMedAI/RareMedData).
- **Knowledge Graph Genetic Variation Data**:
  - **ClinVar**: Download the GRCh38 VCF release (2025.02.09) from the [NCBI FTP server](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/).
  - **HGMD**: Human Gene Mutation Database (2024.02 version) can be accessed via [HGMD](https://www.hgmd.cf.ac.uk/ac/index.php).
> **Data Organization:** Place downloaded graph sources in `RareMed-KG/data/raw/`, clinical inputs in `data/private/`, and model weights in `models/RareSeek-R1/`. Patient records and licensed HGMD content are supplied separately.

---

<a id="getting-started"></a>
## 🚀 Getting Started

### Environment Setup

**Hardware Requirements:** 

- **GPU**: NVIDIA H800 GPUs (Tested primarily on 4 GPUs for parallel inference).
- **OS**: Linux

**Dependencies:**

1. Clone the repository:
   ```bash
   git clone https://github.com/TaoMedAI/RareSeek-R1.git
   cd RareSeek-R1
   ```
2. Create and activate a Conda environment:
   ```bash
   conda create -n rareseek python=3.10
   conda activate rareseek
   ```
3. Install the required packages (requires `PyTorch=2.6.0`, `CUDA=12.6`, `vLLM=0.8.4`):
   ```bash
   pip install -r requirements.txt
   ```

### Model Weights

The model weights of **RareSeek-R1** can be accessed upon request via [HuggingFace (TaoMedAI/RareSeek-R1)](https://huggingface.co/TaoMedAI/RareSeek-R1) for non-commercial research and academic use. 

> **Note:** Once access is granted, please download the model weights and place them in the `models/RareSeek-R1/` directory. You will typically receive a response within one week of submitting your request. If you do not hear back in a timely manner, please contact the corresponding author.

---

<a id="usage-pipeline"></a>
## 🛠 Usage Pipeline

The following sections describe the available components. Standard inference uses EHR narratives directly; phenotype extraction supports phenotype-based comparisons and graph retrieval. Select the components appropriate to your experiment.

Run commands from the repository root. The supplied input is `sample_data/sample.jsonl`, with one JSON record per line:

| Field | Purpose |
| :--- | :--- |
| `id` | Case identifier; numeric and string IDs are supported |
| `context` | Clinical narrative used for inference |
| `disease` | Reference disease name; excluded from model prompts |
| `orphacode` | Reference Orphanet identifier used for evaluation; excluded from model prompts |

No field renaming is required. RAG additionally reads positive `hpo_list` entries and optional `variants`.

### 1. EHR Preprocessing

To standardize multilingual clinical notes, non-English EHRs can be translated into English. Records with `language=en` bypass translation. The supplied sample is English and can be used directly. For other input languages, set `--language` accordingly (for example, `zh`).

```bash
pip install -r requirements/services.txt
export DEEPSEEK_API_KEY="YOUR_KEY"
python rare_disease_ehr_translator.py --input sample_data/sample.jsonl --output outputs/preprocessing/sample_en.jsonl --language en
```

Translation and HPO extraction use their configured backend; only send records authorized for that destination.

### 2. Phenotype Extraction (Baselines & LLMs)

For comparative analysis of phenotype-based approaches and to evaluate the discrepancy between EHR narrative texts and phenotype diagnostics, we employ several extraction algorithms.

#### A. Traditional Baselines
We benchmarked against [PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger) and [PhenoBERT](https://github.com/EclipseCN/PhenoBERT). Please configure the parameters as follows:

| Algorithm | Parameter | Value |
| :--- | :--- | :--- |
| **PhenoTagger** | `model_type` | `pubmedbert` |
| | `onlyLongest` | `True` |
| | `abbrRecog` | `True` |
| | `negation` | `True` |
| | `ML_Threshold` | `0.95` |
| **PhenoBERT** | `param1` | `0.8` |
| | `param2` | `0.6` |
| | `param3` | `0.9` |
| | `use_longest` | `True` |
| | `use_step_3` | `True` |

#### B. LLM-based Phenotype (HPO) Extraction

1. **Model Preparation:** Download [BioLORD-2023-M](https://huggingface.co/FremyCompany/BioLORD-2023-M) for phenotypic terminology embeddings and place it in `BioLORD/`.
2. **LLM Backend:** Deploy [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) locally or use a compatible endpoint, such as [OpenAI](https://platform.openai.com/) or [DeepSeek APIs](https://platform.deepseek.com/). The service defaults to DeepSeek; configure `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL` for your backend.
3. **Start the Extraction Service:**

   ```bash
   uvicorn llm_hpo_extraction_service:app --host 127.0.0.1 --port 8000
   ```

4. **Run the Extraction Client** in another terminal:

   ```bash
   python llm_hpo_api_client.py --input sample_data/sample.jsonl --output outputs/preprocessing/sample_hpo.jsonl
   ```

The command writes `outputs/preprocessing/sample_hpo.jsonl`, retaining extraction details and positive mapped HPO IDs in `hpo_list`.

### 3. Model Inference (Standard)

We use `vLLM` to accelerate standard inference. Clinical records and local weights can be supplied directly; phenotype extraction, reference labels and a knowledge graph are not prerequisites.

**Python Inference:**

```bash
python inference.py --input sample_data/sample.jsonl --model models/RareSeek-R1 --output outputs/standard
```

Predictions are saved to `outputs/standard/predictions.jsonl`. Add `--tensor-parallel-size 4` for four-GPU inference.

**Model Service (via Slurm):**

```bash
sbatch inference/inference.sh
```

The service uses four GPUs on node `gpu3`, partition `llm`, and listens on port `8000`. Adjust the node and partition for your cluster. If running the HPO service on the same machine, use a separate port for one service.

**Batch Inference (via Slurm):**

```bash
sbatch inference/batch.sh --input sample_data/sample.jsonl --model models/RareSeek-R1 --output outputs/standard --tensor-parallel-size 4
```

### 4. RareMedRAG

Beyond standard standalone inference, **RareMedRAG** retrieves biomedical knowledge from **RareMedKG**, including genetic variation and phenotype associations, to support diagnostic reasoning.

#### A. Knowledge Graph Preparation

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

For licensed HGMD data, continue with:

```bash
python RareMed-KG/integration_clinvar_hgmd/format_hgmd.py
python RareMed-KG/integration_clinvar_hgmd/map_hgmd_coords.py
python RareMed-KG/integration_clinvar_hgmd/process_hgmd_annotation.py
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

#### B. RAG Inference

Supply clinical records containing positive HPO IDs and candidate variants:

```bash
python RareMedRAG.py --input outputs/preprocessing/sample_hpo.jsonl --output outputs/rag/raw_predictions.jsonl --model models/RareSeek-R1 --revision YOUR_WEIGHT_VERSION --ic-source data/private/HPO_phenotype.json --tensor-parallel-size 4
```

All input variants and their unique genes are queried. The model receives up to **300 paths**, prioritizing **variant paths**, then **gene paths**, then **phenotype paths ranked by descending mean IC**. IC is computed as `−ln(n/N)` without smoothing.

Variant records use `Gene.refGene`, `AAChange.refGene` (or `hgvs.c`), `Chr`, `Start`, `End`, `Ref` and `Alt`. The IC snapshot must match the imported graph. Use `--path-limit`, `--max-model-len` and `--max-tokens` to set retrieval and context limits.

### 5. Post-processing & Normalization

Extract disease names from the final ranked output and map them to standard identifiers:

1. **Entity Extraction:**

   ```bash
   python rare_disease_entity_extractor.py --input outputs/rag/raw_predictions.jsonl --output outputs/rag/extracted.jsonl
   ```

2. **Standardization:** Match disease names and aliases against the local Orphanet/OMIM graph snapshot.

   ```bash
   python scripts/normalize_diseases.py --input outputs/rag/extracted.jsonl --output outputs/rag/predictions.jsonl --graph data/private/graph.json
   ```

   For optional Monarch Initiative lookups, use `python monarch_api_client.py --query "DISEASE_NAME"` and review the returned candidates.

For standard inference, replace the input path with `outputs/standard/predictions.jsonl` and use corresponding output paths.

### 6. Evaluation

Evaluation supports both **Exact Evaluation** and **Hierarchical Evaluation**.

- **Reference labels:** Use `sample_data/sample.jsonl`; the evaluator reads `id` and `orphacode`. Separate reference files with `sample_id` and a `true_label` list are also supported.
- **Notebook:** Open `evaluation.ipynb` and set the input paths.
- **Command line:**

  ```bash
  python scripts/evaluate.py --gold sample_data/sample.jsonl --predictions outputs/rag/predictions.jsonl --hierarchy data/private/hierarchy.json --output outputs/rag/metrics.json
  ```

Both metrics use the full reference cohort, including missing and empty predictions. Hierarchical matching includes direct parents, children and grandchildren for ORPHA labels. To use the supplied `Classifications of rare diseases.json`, run:

```bash
python scripts/convert_hierarchy.py --output data/private/hierarchy.json
```

---

<a id="code-reference"></a>
## 🧭 Code Reference

| Location | Purpose |
| :--- | :--- |
| `inference.py` / `RareMedRAG.py` | Standard inference / graph-augmented inference |
| `RareMed-KG/` | Source integration and graph construction |
| `rareseek/` | Shared file handling, IC, evidence formatting and evaluation |
| `scripts/` | Graph import, normalization and evaluation commands |
| `scripts/run_three_hop.py` | Combined graph construction, import, RAG and evaluation; see `--help` |

---

<a id="license"></a>
## 📜 License & Contact

- **License**: The code and model weights are licensed for **non-commercial research and academic use only**. 
- **Contact**: For academic collaborations, dataset inquiries, or issues regarding model access, please refer to the corresponding author listed in our paper or open an issue in this repository.
