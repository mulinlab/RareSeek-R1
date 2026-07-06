# RareSeek-R1: Domain-Specialized Large Language Model for Rare-Disease Diagnostic Reasoning

[![Model Weights](https://img.shields.io/badge/Model_Weights-Huggingface-FFD21E.svg)](https://huggingface.co/TaoMedAI/RareSeek-R1)
[![Dataset](https://img.shields.io/badge/Dataset-RareMedData-blue.svg)](https://huggingface.co/datasets/TaoMedAI/RareMedData)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-Non_Commercial-red.svg)](#license)

**RareSeek-R1** is a domain-specialized large language model tailored for rare-disease diagnostic reasoning, developed through a Progressive Parameter-Efficient Transfer Learning framework. 

The model is first instruction-tuned on the clinically grounded **RareMed-Corpus**, a large, multi-source dataset deeply integrated from medical textbooks, guidelines, biomedical literature, and real-world EHR (Electronic Health Record) narratives. It is subsequently fine-tuned on **RareMed-CoT**, a high-fidelity corpus designed to instill explicit, stepwise clinical reasoning that aligns with real-world diagnostic workflows.

![Overview of RareSeek-R1](https://github.com/TaoMedAI/RareSeek-R1/blob/main/RareSeek-R1.png)
*Figure 1: Overall framework and pipeline of RareSeek-R1.*

---

## 📑 Table of Contents
- [Data Availability](#-data-availability)
- [Getting Started](#-getting-started)
  - [Environment Setup](#environment-setup)
  - [Model Weights](#model-weights)
- [Usage Pipeline](#-usage-pipeline)
  - [1. EHR Preprocessing](#1-ehr-preprocessing)
  - [2. Phenotype Extraction](#2-phenotype-extraction-baselines--llms)
  - [3. Model Inference (Standard)](#3-model-inference-standard)
  - [4. RareMedRAG](#4-raremedkg-augmented-inference-kg-rag)
  - [5. Post-processing & Normalization](#5-post-processing--normalization)
  - [6. Evaluation](#6-evaluation)
- [License & Contact](#-license--contact)

---

## 📊 Data Availability

The open-source data underpinning this study, including training datasets and supplementary knowledge graph components, are publicly available.

- **RareMedData (Training Data & General KG)**: Available at [HuggingFace: TaoMedAI/RareMedData](https://huggingface.co/datasets/TaoMedAI/RareMedData).
- **Knowledge Graph Genetic Variation Data**:
  - **ClinVar**: Download the GRCh38 VCF release (2025.02.09) from the [NCBI FTP server](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/).
  - **HGMD**: Human Gene Mutation Database (2024.02 version) can be accessed via [HGMD](https://www.hgmd.cf.ac.uk/ac/index.php).
> **Data Organization:** We suggest organizing all downloaded data files within the `RareMed-KG/data/raw` directory.
---

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

> **Note:** Once access is granted, please download the model weights and place them in the `models/` directory. You will typically receive a response within one week of submitting your request. If you do not hear back in a timely manner, please contact the corresponding author.

---

## 🛠 Usage Pipeline

Our framework supports a comprehensive end-to-end pipeline, from EHR preprocessing to diagnostic evaluation.

### 1. EHR Preprocessing
To standardize multi-lingual real-world clinical notes, we translate non-English (e.g., Chinese) EHRs into English. English EHRs will be automatically bypassed during this step.
```bash
python rare_disease_ehr_translator.py
```

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
We utilize LLMs for high-fidelity Human Phenotype Ontology (HPO) extraction. 
1. **Model Preparation**: Download [BioLORD-2023-M](https://huggingface.co/FremyCompany/BioLORD-2023-M) for phenotypic terminology embeddings and place it in the `BioLORD/` folder.
2. **LLM Backend**: You can either deploy [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) locally or use API calls (e.g., [OpenAI](https://platform.openai.com/) / [DeepSeek APIs](https://platform.deepseek.com/)).
3. **Start the Extraction Service**:
   ```bash
   uvicorn llm_hpo_extraction_service:app --host 0.0.0.0 --port 8000
   ```
4. **Run the Extraction Client** (Example using DeepSeek-R1 API):
   ```bash
   python llm_hpo_api_client.py
   ```

### 3. Model Inference (Standard)
We utilize `vLLM` to accelerate standard inference. You can deploy the model in a batch setting or run it interactively.

**Batch Inference (via Slurm):**
```bash
cd inference
sbatch inference.sh
```

**Python Demo Inference:**
```python
import json
from vllm import LLM, SamplingParams

def load_data(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line.strip()) for line in file]

def build_prompt(context):
    return f"{context} What is the rare disease that the patient is most likely to be diagnosed with?"

# 1. Load sample data
data = load_data("sample_data/sample.jsonl")

# 2. Configure sampling parameters
sampling_params = SamplingParams(temperature=0.1, top_p=0.95)

# 3. Initialize vLLM (Replace with local path "models/RareSeek-R1" if downloaded)
llm = LLM(model="TaoMedAI/RareSeek-R1") 

# 4. Generate predictions
for entry in data:
    prompt = build_prompt(entry["context"])
    outputs = llm.generate([prompt], sampling_params)
    
    for output in outputs:
        print(f"Prompt: {prompt}\nGenerated Text: {output.outputs[0].text}\n{'-'*50}")
```

### 4. RareMedRAG
Beyond standard standalone inference, our framework supports **RareMedRAG**. By integrating the **RareMedKG** database, this module retrieves relevant external biomedical knowledge (such as specific genetic variations and deep phenotypic associations) to dynamically enhance the model's diagnostic accuracy and reasoning capability.
```bash
python RareMedRAG.py
```

### 5. Post-processing & Normalization
After generating the narrative diagnostic reasoning, we extract and standardize the disease entities.

1. **Entity Extraction**: Extract specific disease names from the LLM outputs.
   ```bash
   python rare_disease_entity_extractor.py
   ```
2. **Standardization**: Normalize the extracted disease entities into standard vocabularies (Orphanet and OMIM codes) via the Monarch Initiative API.
   ```bash
   python monarch_api_client.py
   ```

### 6. Evaluation
We provide comprehensive evaluation scripts to measure the diagnostic accuracy using both **Exact Evaluation** and **Hierarchical Evaluation**.

- **Reference Format**: See the standardized output format at `./sample_data/EHR-Internal_Phenotype.jsonl`.
- **Run Evaluation**: Execute `evaluation.ipynb` in Jupyter Notebook.
  > *Note:* The Hierarchical Evaluation relies on the Orphanet hierarchical structure file: `Classifications of rare diseases.json`. Ensure this file is present in your working directory.

---

## 📜 License & Contact

- **License**: The code and model weights are licensed for **non-commercial research and academic use only**. 
- **Contact**: For academic collaborations, dataset inquiries, or issues regarding model access, please refer to the corresponding author listed in our paper or open an issue in this repository.
