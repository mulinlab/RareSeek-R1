import logging
import requests
from typing import List

# Configure logging for professional/academic standard output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fetch_disease_identifiers(phenotype_term: str) -> List[str]:
    """
    Queries the Monarch Initiative (MONDO) API to retrieve validated Orphanet 
    and OMIM cross-references for a specified disease phenotype.

    Args:
        phenotype_term (str): The clinical or academic designation of the disease.

    Returns:
        List[str]: A list of extracted, standardized database identifiers (e.g., ORPHA, OMIM).
                   Returns an empty list if no corresponding identifiers are resolved.
    """
    api_endpoint = "https://api-v3.monarchinitiative.org/v3/api/search"
    
    query_parameters = {
        "q": phenotype_term,
        "limit": 20,
        "offset": 0
    }
    
    request_headers = {
        "accept": "application/json"
    }
    
    try:
        # A timeout is enforced to prevent hanging requests in production environments
        response = requests.get(
            api_endpoint, 
            headers=request_headers, 
            params=query_parameters,
            timeout=15
        )
        response.raise_for_status()
        
        response_payload = response.json()
        
        # Iterate through the returned ontology entities
        for ontology_entity in response_payload.get("items", []):
            cross_references = ontology_entity.get("xref")
            
            if cross_references and isinstance(cross_references, list):
                # Initialize an accumulator for the requisite target identifiers
                target_identifiers = []
                
                for xref in cross_references:
                    if xref.startswith("Orphanet:"):
                        # Reformat the prefix to comply with standard ORPHA nomenclature
                        target_identifiers.append(f"ORPHA:{xref.split(':')[1]}")
                    elif xref.startswith("OMIM:"):
                        # OMIM prefix is retained in its native format
                        target_identifiers.append(xref)
                
                # Return identifiers from the highest-ranked mapped entity immediately
                if target_identifiers:
                    return target_identifiers
                    
    except requests.exceptions.RequestException as error:
        logging.error(f"API request failed during the query for '{phenotype_term}'. Details: {error}")
        
    return []

# Demonstration of algorithmic functionality
if __name__ == "__main__":
    sample_query = "GRANULOMATOSIS WITH POLYANGIITIS"
    extracted_identifiers = fetch_disease_identifiers(sample_query)
    
    if extracted_identifiers:
        logging.info(f"Identifiers successfully retrieved for '{sample_query}': {extracted_identifiers}")
    else:
        logging.warning(f"No identifiers could be mapped for '{sample_query}'.")