import logging
import os
import re
import requests
from typing import List

# Configure logging for professional/academic standard output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fetch_disease_identifiers(phenotype_term: str) -> List[str]:
    """
    Queries the Monarch Initiative (MONDO) API to retrieve candidate Orphanet 
    and OMIM cross-references for a specified disease phenotype.

    Args:
        phenotype_term (str): The clinical or academic designation of the disease.

    Returns:
        List[str]: A list of extracted, standardized database identifiers (e.g., ORPHA, OMIM).
                   Returns an empty list if no corresponding identifiers are resolved.
    """
    api_endpoint = os.getenv("MONARCH_API_URL", "https://api-v3.monarchinitiative.org/v3/api").rstrip("/") + "/search"
    
    query_parameters = {
        "q": phenotype_term,
        "limit": 20,
        "offset": 0,
        "category": "biolink:Disease"
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
            category = ontology_entity.get('category', [])
            category = [category] if isinstance(category, str) else category
            if category and 'biolink:Disease' not in category:
                continue
            cross_references = ontology_entity.get("xref", []) or []
            cross_references = [cross_references] if isinstance(cross_references, str) else cross_references
            if not isinstance(cross_references, list):
                cross_references = []
            if isinstance(ontology_entity.get('id'), str):
                cross_references = [*cross_references, ontology_entity['id']]
            
            if cross_references and isinstance(cross_references, list):
                # Initialize an accumulator for the requisite target identifiers
                target_identifiers = []
                
                for xref in cross_references:
                    if not isinstance(xref, str):
                        continue
                    if re.fullmatch(r'(?:Orphanet|ORPHA):[0-9]+', xref):
                        # Reformat the prefix to comply with standard ORPHA nomenclature
                        target_identifiers.append(f"ORPHA:{xref.split(':')[1]}")
                    elif re.fullmatch(r'OMIM:[0-9]+', xref):
                        # OMIM prefix is retained in its native format
                        target_identifiers.append(xref)
                
                # Return identifiers from the highest-ranked mapped entity immediately
                if target_identifiers:
                    return sorted(set(target_identifiers))
                    
    except requests.exceptions.RequestException as error:
        raise RuntimeError('Monarch lookup failed') from error
        
    return []

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Query Monarch for candidate disease identifiers")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    print(json.dumps(fetch_disease_identifiers(args.query), ensure_ascii=False))
