# RareSeek-R1

**RareSeek-R1** is a domain-specialized large language model for rare-disease diagnostic reasoning, developed through a Progressive Parameter-Efficient Transfer Learning framework. The model is first instruction-tuned on the clinically grounded RareMed-Corpus, a large, multi-source dataset deeply integrated from medical textbooks, guidelines, biomedical literature, and real-world EHR narratives. It is then fine-tuned on RareMed-CoT, a high-fidelity corpus designed to instill explicit, stepwise clinical reasoning aligned with real diagnostic workflows.

For an overview of this study, see the figure below.

![Fig1](https://github.com/TaoMedAI/RareSeek-R1/blob/main/RareSeek-R1.png)

## Requirements
### Hardware
- **GPU**: NVIDIA H800 GPUs (tested on 4 GPUs for inference)
### Dependencies
- Linux
- Python 3.10
- PyTorch = 2.6.0
- CUDA = 12.6
- transformers = 4.51.3
- tokenizers = 0.21.1
- vLLM = 0.8.4

## Installation
```
$ pip install -r requirements.txt
```

## Model weights
The pretrained weights of RareSeek-R1 can be accessed upon request via [Huggingface](https://huggingface.co/TaoMedAI/RareSeek-R1) for non-commercial research and academic use. Once granted access, please download the model weights and place them in the **models** folder.
You will typically receive a response within one week of submitting your request. If you do not hear back in a timely manner, please contact the corresponding author listed in the paper.

## Inference

### Installation

1. **Clone the repository**:
   ```python
   git clone <https://github.com/TaoMedAI/RareSeek-R1>
   ```
  
2. **Create conda environment**:
   ```python
   conda create -n rareseek python=3.10
   conda activate rareseek
   ```
   
3. **Install dependencies**:
   ```python
   pip install -r requirements.txt
   ```

### Deploy vLLM for inference

```python
cd inference
sbatch inference.sh
```

### Run demo
```python
import json
from vllm import LLM, SamplingParams

# Function to load data from the JSONL file
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Function to build the prompt by appending a question to the context
def build_prompt(context):
    return f"{context} What is the rare disease that the patient is most likely to be diagnosed with?"

# Load the data from the sample.jsonl file
file_path = "sample_data/sample.jsonl"  # Path to your jsonl file
data = load_data(file_path)

# Create sampling parameters for controlling the model output
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Specify the model path (replace with local path if downloaded)
# If you have the model downloaded, replace "TaoMedAI/RareSeek-R1" with the local path to the model directory
llm = LLM(model="TaoMedAI/RareSeek-R1")  # Use local path if model is downloaded, e.g. "path/to/model"

# Iterate over the data and generate predictions
for entry in data:
    context = entry["context"]
    prompt = build_prompt(context)
    
    # Use the LLM to generate text based on the prompt
    outputs = llm.generate([prompt], sampling_params)
    
    # Print the results
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated Text: {generated_text!r}")
        print("-" * 50)
```
