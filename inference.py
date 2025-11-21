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
