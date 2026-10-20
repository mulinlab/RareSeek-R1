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
# Command-line interface
# ==========================================
if __name__ == "__main__":
    import argparse
    from rareseek.io import path, read_jsonl, write_jsonl, by_id
    p = argparse.ArgumentParser(description="Batch HPO extraction; sends context to the configured service")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--url", default="http://localhost:8000")
    a = p.parse_args()
    rows = read_jsonl(path(a.input))
    by_id(rows)
    client = PhenotypeAPIClient(a.url)
    for row in rows:
        response = client.extract_single_record(row["context"])
        if response is None or response.get("status") != "success":
            raise RuntimeError("HPO service failed")
        if not isinstance(response.get('data'), list) or any(not isinstance(item, dict) for item in response['data']):
            raise ValueError('HPO service returned a malformed result')
        row["phenotype_extraction"] = response["data"]
        row["hpo_list"] = sorted({x["Mapped_HPO_ID"] for x in response["data"]
            if x.get("Mapped_HPO_ID") and str(x.get("Negation_Status", "")).strip().lower() == "positive"})
    write_jsonl(path(a.output), rows)
