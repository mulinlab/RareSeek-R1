import os
import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI, OpenAIError

# ==========================================
# 1. Configuration & Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("RareDiseaseEntityExtractor")

# ==========================================
# 2. Core Extractor Class
# ==========================================
class RareDiseaseEntityExtractor:
    """
    A professional pipeline component for extracting rare disease ontology entities 
    from diagnostic reasoning texts (LLM outputs) using OpenAI's GPT-4o.
    """
    
    SYSTEM_PROMPT = """
    You are an expert in rare disease ontology curation.
    Your task is to extract all disease names mentioned in the following diagnostic reasoning text.
    
    Requirements:
    1. Identify only disease or syndrome names (including rare or complex conditions).
    2. Ignore reasoning, modifiers, or adjectives not part of the disease name.
    3. If multiple diseases are mentioned, list them all.
    4. Output strictly in JSON format as:
    {"diseases": ["Disease Name 1", "Disease Name 2", ...]}
    """

    def __init__(self, model_name: str = "gpt-4o") -> None:
        """
        Initializes the OpenAI client and extractor configurations.
        
        Args:
            model_name (str): The specific OpenAI model to utilize (default: gpt-4o).
        """
        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.error("OPENAI_API_KEY environment variable is missing.")
            raise ValueError("Authentication requires OPENAI_API_KEY to be set in the environment.")
            
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized entity extractor with model: {self.model_name}")

    def extract_disease_entities(self, diagnostic_text: str) -> List[str]:
        """
        Invokes the LLM to extract disease entities from the provided diagnostic text.
        
        Args:
            diagnostic_text (str): The raw diagnostic reasoning text from predict_result.
            
        Returns:
            List[str]: A list of extracted standardized disease names.
        """
        if not diagnostic_text or not diagnostic_text.strip():
            logger.warning("Empty diagnostic text provided. Returning empty list.")
            return []

        prompt_content = f"Text:\n<<<\n{diagnostic_text}\n>>>"

        try:
            # Utilizing Standard Chat Completions API with JSON Mode enforced
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # Low temperature for highly deterministic analytical tasks
            )
            
            raw_output = response.choices[0].message.content
            
            if not raw_output:
                return []
                
            parsed_json = json.loads(raw_output)
            extracted_diseases = parsed_json.get("diseases", [])
            
            return extracted_diseases
            
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse LLM output into JSON: {json_err}")
            return []
        except OpenAIError as api_err:
            logger.error(f"OpenAI API error occurred during extraction: {api_err}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return []

# ==========================================
# 3. Batch Processing Logic
# ==========================================
def process_diagnostic_dataset(input_filepath: str, output_filepath: str) -> None:
    """
    Reads a JSONL file, extracts disease entities from the 'predict_result' field,
    and writes the enriched data to an output JSONL file.
    
    Args:
        input_filepath (str): Path to the source JSONL file (e.g., RareSeek-R1.jsonl).
        output_filepath (str): Path where the processed JSONL file will be saved.
    """
    extractor = RareDiseaseEntityExtractor(model_name="gpt-4o")
    
    if not os.path.exists(input_filepath):
        logger.error(f"Input dataset not found: {input_filepath}")
        return

    processed_count = 0
    logger.info(f"Initiating batch processing for dataset: {input_filepath}")

    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue
                
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Line {line_number}: Invalid JSON structure. Skipping record.")
                continue

            # Target the specific diagnostic key
            diagnostic_reasoning = record.get("predict_result", "")
            
            if diagnostic_reasoning:
                logger.info(f"Processing clinical record at line {line_number}...")
                extracted_entities = extractor.extract_disease_entities(diagnostic_reasoning)
                
                # Append the newly extracted data back into the original record structure
                record["extracted_diseases"] = extracted_entities
            else:
                logger.warning(f"Line {line_number}: 'predict_result' key missing or empty.")
                record["extracted_diseases"] = []

            # Write the updated record to the target JSONL file
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_count += 1

    logger.info(f"Batch processing pipeline complete. Successfully processed {processed_count} records.")
    logger.info(f"Enriched dataset saved to: {output_filepath}")

# ==========================================
# 4. Execution Entry Point
# ==========================================
if __name__ == "__main__":
    # Ensure you have your OpenAI API key exported before running:
    # Example (Linux/macOS): export OPENAI_API_KEY="sk-..."
    # Example (Windows CMD): set OPENAI_API_KEY="sk-..."
    
    INPUT_DATASET = "RareSeek-R1.jsonl"
    OUTPUT_DATASET = "RareSeek-R1_extracted.jsonl"
    
    # Execute the pipeline
    process_diagnostic_dataset(
        input_filepath=INPUT_DATASET,
        output_filepath=OUTPUT_DATASET
    )