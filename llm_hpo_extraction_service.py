import os
import json
import re
import asyncio
import logging
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI, OpenAIError
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 1. Configuration & Global Variables
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your_api_key_here")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Asynchronous LLM Client
llm_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# Global Model and Data References
embedding_model: Optional[SentenceTransformer] = None
hpo_embeddings_tensor: Optional[torch.Tensor] = None
hpo_identifier_list: List[str] = []
hpo_term_list: List[str] = []
hpo_knowledge_base: Dict[str, Dict[str, str]] = {}

# ==========================================
# 2. Application Lifecycle Management
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the FastAPI application.
    Pre-loads the embedding model and phenotype ontology into memory upon startup.
    """
    global embedding_model, hpo_embeddings_tensor
    global hpo_identifier_list, hpo_term_list, hpo_knowledge_base
    
    logger.info("Step 1/3: Loading BioLORD-2023-M embedding model...")
    try:
        embedding_model = SentenceTransformer('./BioLORD-2023-M/')
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise RuntimeError("Embedding model initialization failed.")

    logger.info("Step 2/3: Loading HPO knowledge base...")
    try:
        with open('zh-HPO.json', 'r', encoding='utf-8') as file:
            hpo_knowledge_base = json.load(file)
    except FileNotFoundError:
        logger.warning("Ontology file 'zh-HPO.json' not found. Utilizing synthetic fallback data.")
        hpo_knowledge_base = {
            "HP:0000001": {"hpo_term": "All", "chpo_term": "All"},
            "HP:0000002": {"hpo_term": "Abnormality of body height", "chpo_term": "Abnormality of body height"}
        }
        
    hpo_identifier_list = list(hpo_knowledge_base.keys())
    hpo_term_list = [hpo_knowledge_base[h_id]['hpo_term'] for h_id in hpo_identifier_list]
    
    logger.info("Step 3/3: Vectorizing HPO terms into global tensor...")
    hpo_embeddings_tensor = embedding_model.encode(hpo_term_list, convert_to_tensor=True)
    
    logger.info("Initialization complete. API is ready to accept connections.")
    
    yield  # Application runtime
    
    logger.info("Initiating service shutdown sequence...")

# Initialize FastAPI Router
app = FastAPI(
    lifespan=lifespan, 
    title="Clinical Phenotype Extraction API",
    description="Asynchronous API for extracting and standardizing clinical phenotypes from EHRs using LLMs and BioLORD."
)

# ==========================================
# 3. Data Transfer Objects (DTOs)
# ==========================================
class ClinicalRecordRequest(BaseModel):
    text: str = Field(..., description="The raw clinical text of the patient.")

class BatchClinicalRecordRequest(BaseModel):
    texts: List[str] = Field(..., description="A list of raw clinical texts for batch processing.")

# ==========================================
# 4. Core Processing Logic
# ==========================================
async def extract_clinical_phenotypes(patient_text: str) -> List[Dict[str, str]]:
    """
    Interfaces with the LLM to extract structured phenotypic data.
    """
    extraction_prompt = f"""
    You are a clinical NLP specialist focused on structured extraction of phenotypic information from electronic health records (EHR). Your task is to extract clinically relevant phenotypic findings from the given patient text in a structured way.
    
    ========================
    TASK INSTRUCTIONS
    ========================
    1. Carefully read the patient text.
    2. Identify ALL phenotypic mentions, including:
       - Present / positive findings (symptoms, signs, diagnoses, abnormalities)
       - Explicitly NEGATED findings (e.g., "no fever", "denies seizures")
       - Ruled-out or absent conditions explicitly stated in the text
    3. Do NOT infer new conditions. Only extract what is explicitly stated.
    4. Normalize each phenotype into a formal English clinical description (standard clinical terminology preferred).
    5. Preserve the original wording as it appears in the text.
    6. Assign a negation status for each extracted phenotype:
       - "positive" → phenotype is present / observed
       - "negation" → phenotype is explicitly denied or absent
       
    ========================
    OUTPUT REQUIREMENTS
    ========================
    - Output MUST be a valid JSON array only.
    - Do NOT include markdown, explanations, or additional text.
    - Each item must follow this structure exactly:
    [
      {{
        "Phenotype": "Standardized English clinical phenotype",
        "Original_term": "Exact phrase from patient text",
        "Negation": "positive | negation"
      }}
    ]
    
    ========================
    PATIENT TEXT
    ========================
    {patient_text}
    """
    
    try:
        response = await llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a highly precise medical NLP system."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.1,
            stream=False
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # Sanitize potential markdown artifacts
        sanitized_output = re.sub(r'^```json\s*', '', raw_output, flags=re.MULTILINE)
        sanitized_output = re.sub(r'^```\s*', '', sanitized_output, flags=re.MULTILINE)
        
        return json.loads(sanitized_output)
        
    except OpenAIError as api_error:
        logger.error(f"LLM API request failed: {api_error}")
        return []
    except json.JSONDecodeError as json_error:
        logger.error(f"Failed to parse LLM output into JSON: {json_error}\nRaw Output: {raw_output}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during extraction: {e}")
        return []

def map_phenotypes_to_ontology(extracted_items: List[Dict[str, str]], similarity_threshold: float = 0.80) -> List[Dict[str, Any]]:
    """
    Maps extracted raw phenotypes to standard HPO identifiers using cosine similarity.
    """
    normalized_results = []
    if not extracted_items:
        return normalized_results

    # Isolate phenotypes for batch vectorization
    query_phenotypes = [item.get('Phenotype', '') for item in extracted_items if item.get('Phenotype')]
    
    if not query_phenotypes:
        return normalized_results

    # Perform batch tensor encoding
    query_embeddings_tensor = embedding_model.encode(query_phenotypes, convert_to_tensor=True)
    
    # Compute cosine similarity matrix
    cosine_similarity_matrix = util.cos_sim(query_embeddings_tensor, hpo_embeddings_tensor)
    
    # Extract maximum scores and corresponding indices
    max_scores, optimal_indices = torch.max(cosine_similarity_matrix, dim=1)
    
    for iteration_idx, source_item in enumerate(extracted_items):
        if not source_item.get('Phenotype'):
            continue
            
        similarity_score = max_scores[iteration_idx].item()
        ontology_idx = optimal_indices[iteration_idx].item()
        
        mapped_entity = {
            'Original_Term': source_item.get('Original_term', ''),
            'Extracted_Phenotype': source_item.get('Phenotype', ''),
            'Negation_Status': source_item.get('Negation', 'positive'),
            'Similarity_Score': round(similarity_score, 4)
        }

        if similarity_score >= similarity_threshold:
            best_match_id = hpo_identifier_list[ontology_idx]
            mapped_entity.update({
                'Mapped_HPO_ID': best_match_id,
                'Mapped_HPO_Term': hpo_term_list[ontology_idx],
                'Mapped_Local_Term': hpo_knowledge_base[best_match_id].get('chpo_term', ''),
                'Status': 'Matched'
            })
        else:
            mapped_entity.update({
                'Mapped_HPO_ID': None,
                'Mapped_HPO_Term': None,
                'Mapped_Local_Term': None,
                'Status': 'Sub-threshold confidence'
            })
            
        normalized_results.append(mapped_entity)
        
    return normalized_results

# ==========================================
# 5. API Endpoints
# ==========================================
@app.post("/api/v1/extract", response_model=Dict[str, Any])
async def process_single_record(record: ClinicalRecordRequest):
    """
    Extracts and maps phenotypes from a single clinical patient record.
    """
    extracted_data = await extract_clinical_phenotypes(record.text)
    ontology_mapped_data = map_phenotypes_to_ontology(extracted_data)
    
    return {
        "status": "success", 
        "data": ontology_mapped_data
    }

@app.post("/api/v1/extract-batch", response_model=Dict[str, Any])
async def process_batch_records(records: BatchClinicalRecordRequest):
    """
    Concurrently processes multiple clinical records to optimize network latency.
    """
    extraction_tasks = [extract_clinical_phenotypes(text) for text in records.texts]
    batch_extracted_data = await asyncio.gather(*extraction_tasks)
    
    batch_mapped_data = [
        map_phenotypes_to_ontology(extracted_record) 
        for extracted_record in batch_extracted_data
    ]
    
    return {
        "status": "success", 
        "batch_data": batch_mapped_data
    }

if __name__ == "__main__":
    import uvicorn
    # Execute the server
    uvicorn.run("llm_hpo_extraction_service:app", host="0.0.0.0", port=8000, reload=False)