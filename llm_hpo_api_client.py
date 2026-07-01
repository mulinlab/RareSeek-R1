import json
import logging
import requests
from typing import Dict, Any, List, Optional

# Configure logging for the API client
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class PhenotypeAPIClient:
    """
    A professional HTTP client for interacting with the Clinical Phenotype Extraction API.
    Handles single and batch record processing for downstream phenotype mapping.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initializes the API client.
        
        Args:
            base_url (str): The base URL of the FastAPI server.
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def extract_single_record(self, clinical_text: str) -> Optional[Dict[str, Any]]:
        """
        Sends a single clinical record to the API for phenotype extraction and HPO mapping.
        """
        endpoint = f"{self.base_url}/api/v1/extract"
        payload = {"text": clinical_text}

        logger.info(f"Sending single record extraction request to {endpoint}")
        
        try:
            response = requests.post(endpoint, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request failed during single extraction: {e}")
            return None

    def extract_batch_records(self, clinical_texts: List[str]) -> Optional[Dict[str, Any]]:
        """
        Sends multiple clinical records to the API for concurrent processing.
        """
        endpoint = f"{self.base_url}/api/v1/extract-batch"
        payload = {"texts": clinical_texts}

        logger.info(f"Sending batch extraction request ({len(clinical_texts)} records) to {endpoint}")
        
        try:
            response = requests.post(endpoint, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request failed during batch extraction: {e}")
            return None

# ==========================================
# Execution & Testing
# ==========================================
if __name__ == "__main__":
    
    # Simulated Rare Disease EHRs (English only)
    # Case 1: Suspected connective tissue or metabolic disorder
    simulated_ehr_01 = """
    Chief Complaint: A 6-year-old male presents with severe global developmental delay and recurrent seizures.
    History of Present Illness: The patient was born at term but exhibited severe generalized hypotonia at birth. 
    Over the past three years, parents noted progressive distal muscle weakness, frequent generalized tonic-clonic 
    seizures, and macrocephaly. The patient denies any visual loss or hearing impairment. No fever, rash, or 
    recurrent respiratory infections observed.
    Physical Examination: Remarkable for frontal bossing, severe joint hypermobility, and bilateral pes planus. 
    Cardiovascular exam reveals normal heart sounds; explicitly, there is no mitral valve prolapse or aortic root dilation. 
    Ophthalmology consult noted bilateral optic atrophy.
    """

    # Case 2: Suspected skeletal dysplasia
    simulated_ehr_02 = """
    Patient is a 12-month-old female evaluated for disproportionate short stature and rhizomelic limb shortening. 
    Radiographic findings confirm achondroplasia with narrow interpedicular distances in the lumbar spine. 
    The patient has no cleft palate and no polydactyly. 
    """

    # Instantiate the client (Ensure your FastAPI server is running on localhost:8000)
    api_client = PhenotypeAPIClient(base_url="http://localhost:8000")

    # ---------------------------------------------------------
    # Test 1: Single Record Extraction
    # ---------------------------------------------------------
    logger.info("--- Initiating Single Record Test ---")
    single_result = api_client.extract_single_record(simulated_ehr_01)
    
    if single_result:
        print("\n" + "="*60)
        print(" SINGLE RECORD EXTRACTION RESULTS ")
        print("="*60)
        # Using json.dumps for pretty printing the JSON output
        print(json.dumps(single_result, indent=4, ensure_ascii=False))
        print("="*60 + "\n")

    # ---------------------------------------------------------
    # Test 2: Batch Record Extraction
    # ---------------------------------------------------------
    logger.info("--- Initiating Batch Record Test ---")
    batch_texts = [simulated_ehr_01, simulated_ehr_02]
    batch_result = api_client.extract_batch_records(batch_texts)
    
    if batch_result:
        print("\n" + "="*60)
        print(" BATCH RECORD EXTRACTION RESULTS ")
        print("="*60)
        print(json.dumps(batch_result, indent=4, ensure_ascii=False))
        print("="*60 + "\n")