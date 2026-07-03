"""
RareMedRAG Pipeline for Rare Disease Diagnosis
"""

import os
import json
import math
import re
import logging
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

from neo4j import GraphDatabase
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ==============================================================================
# 1. System Configuration & Initialization
# ==============================================================================

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model Configuration
MODEL_PATH = "TaoMedAI/RareSeek-R1"

# Neo4j Database Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://0.0.0.0:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

# Graph Retrieval Constraints
HPO_IC_FILE_PATH = os.getenv("HPO_IC_PATH", "HPO_phenotype.json")
MAX_SUBGRAPHS_LIMIT = 300
CYPHER_PATH_LIMIT = 2000

# Input/Output Paths
INPUT_DATA_PATH = "test.jsonl"
OUTPUT_DATA_PATH = "test_RareMedRAG_top20_results.jsonl"

# Initialize Database Driver
db_driver = GraphDatabase.driver(
    NEO4J_URI, 
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD), 
    connection_timeout=300
)

# ==============================================================================
# 2. Utility & Pre-computation Functions
# ==============================================================================

def compute_information_content(file_path: str) -> Dict[str, float]:
    """
    Computes Information Content (IC) for Human Phenotype Ontology (HPO) terms.
    """
    if not os.path.exists(file_path):
        logger.warning(f"HPO data file not found at {file_path}. Returning empty IC dict.")
        return {}
        
    with open(file_path, "r", encoding="utf-8") as file:
        disease_hpo_data = json.load(file)

    hpo_frequency: Dict[str, int] = {}
    total_diseases = max(1, len(disease_hpo_data))

    for info in disease_hpo_data.values():
        hpo_list = info.get("hpo_list", []) if isinstance(info, dict) else []
        for hpo_term in set(hpo_list):
            hpo_frequency[hpo_term] = hpo_frequency.get(hpo_term, 0) + 1

    hpo_ic_scores = {
        hpo: -math.log(count / total_diseases) 
        for hpo, count in hpo_frequency.items()
    }
    return hpo_ic_scores

def load_hpo_terminology(file_path: str = "zh-HPO.json") -> Dict[str, str]:
    """
    Loads mapping of HPO IDs to human-readable terms.
    """
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {k: v.get("hpo_term", k) for k, v in data.items()}

# Global Initializations
HPO_IC_SCORES = compute_information_content(HPO_IC_FILE_PATH)
HPO_TERMINOLOGY = load_hpo_terminology()

# ==============================================================================
# 3. Knowledge Graph Retrieval Modules
# ==============================================================================

def extract_hpo_identifiers(record: Dict[str, Any]) -> List[str]:
    """
    Extracts HPO IDs from various possible fields within a clinical record.
    """
    extracted_hpos = []
    
    if "hpo_list" in record:
        extracted_hpos.extend(record["hpo_list"])
    elif "hpos" in record:
        extracted_hpos.extend(record["hpos"])
    
    # Fallback: Regex extraction from clinical narrative
    clinical_narrative = record.get("trans_result", "")
    regex_matches = re.findall(r"(HP:\d{7})", clinical_narrative)
    extracted_hpos.extend(regex_matches)
    
    return list(set(extracted_hpos))

def retrieve_phenotype_neighborhood(phenotype_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Executes a 3-hop Cypher traversal centered around provided Phenotype nodes.
    """
    if not phenotype_ids:
        return []
        
    cypher_query = """
    UNWIND $phenotypes AS phenotype_id
    MATCH path = (p:Phenotype)-[*1..3]-(related_node)
    WHERE p.hpo_name CONTAINS phenotype_id OR p.HPOId CONTAINS phenotype_id
    WITH path LIMIT $db_limit
    WITH {
        nodes: [n IN nodes(path) | { 
            labels: labels(n), 
            properties: apoc.map.removeKeys(properties(n), ['embedding']) 
        }],
        relationships: [r IN relationships(path) | { type: type(r) }]
    } AS subgraph
    RETURN subgraph
    """
    try:
        with db_driver.session() as session:
            result = session.run(
                cypher_query, 
                phenotypes=phenotype_ids, 
                db_limit=CYPHER_PATH_LIMIT
            )
            return [record.get("subgraph") for record in result if record.get("subgraph")]
    except Exception as error:
        logger.error(f"Phenotype KG Retrieval Exception: {error}")
        return []

def prioritize_subgraphs_by_ic(subgraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ranks extracted subgraphs based on the average Information Content of their Phenotypes.
    """
    scored_subgraphs = []
    
    for subgraph in subgraphs:
        unique_phenotypes = set()
        for node in subgraph.get("nodes", []):
            if "Phenotype" in node.get("labels", []):
                properties = node.get("properties", {})
                hpo_id = properties.get("HPOId") or properties.get("hpo_id")
                if hpo_id:
                    unique_phenotypes.add(str(hpo_id).strip())
        
        ic_values = [HPO_IC_SCORES.get(h, 0.0) for h in unique_phenotypes]
        non_zero_ic = [val for val in ic_values if val > 0]
        average_ic = sum(non_zero_ic) / len(non_zero_ic) if non_zero_ic else 0.0
        
        subgraph_copy = dict(subgraph)
        subgraph_copy["_average_ic"] = average_ic
        scored_subgraphs.append(subgraph_copy)
        
    scored_subgraphs.sort(key=lambda x: x.get("_average_ic", 0.0), reverse=True)
    return scored_subgraphs[:min(MAX_SUBGRAPHS_LIMIT, len(scored_subgraphs))]

def format_phenotype_context(subgraphs: List[Dict[str, Any]]) -> str:
    """
    Structures the prioritized phenotype subgraphs into a textual prompt context.
    Ranks the associated diseases based on the average Information Content (IC) 
    of their matched phenotypes and retains the top 50.
    """
    disease_mapping = []
    
    for subgraph in subgraphs:
        phenotypes, disease_codes, disease_names = [], [], []
        for node in subgraph.get("nodes", []):
            if "Phenotype" in node.get("labels", []):
                phenotypes.append(node.get("properties", {}).get("HPOId", ""))
            if "Disease" in node.get("labels", []):
                disease_codes.append(node.get("properties", {}).get("ORPHAcode", node.get("properties", {}).get("externalReferences", "")))
                disease_names.append(node.get("properties", {}).get("disease_name", "Unknown"))
                
        if phenotypes and disease_codes:
            disease_mapping.append({
                "Phenotypes": phenotypes, 
                "DiseaseCodes": disease_codes, 
                "DiseaseNames": disease_names
            })

    aggregated_disease_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    for entry in disease_mapping:
        phenos = entry["Phenotypes"]
        for code, name in zip(entry["DiseaseCodes"], entry["DiseaseNames"]):
            # Calculate average IC for the matched phenotypes
            ic_values = [HPO_IC_SCORES.get(h, 0.0) for h in phenos]
            average_ic = sum(ic_values) / len(ic_values) if ic_values else 0.0
            
            key = (code, name)
            if key not in aggregated_disease_data:
                aggregated_disease_data[key] = {"avg_ic": average_ic, "phenotypes": set(phenos)}
            else:
                # Retain the maximum average IC score if the disease is encountered multiple times
                aggregated_disease_data[key]["avg_ic"] = max(aggregated_disease_data[key]["avg_ic"], average_ic)
                aggregated_disease_data[key]["phenotypes"].update(phenos)

    # Sort diseases by average IC in descending order and keep the Top 50
    top_diseases = sorted(
        [
            {"name": k[1], "avg_ic": v["avg_ic"], "phenotypes": list(v["phenotypes"])} 
            for k, v in aggregated_disease_data.items()
        ], 
        key=lambda x: x["avg_ic"], 
        reverse=True
    )[:50]

    context_lines = []
    for index, disease_info in enumerate(top_diseases, start=1):
        readable_phenotypes = [HPO_TERMINOLOGY.get(h, h) for h in disease_info["phenotypes"]]
        context_lines.append(f"  {index}. {disease_info['name']} - Associated Phenotypes: {', '.join(readable_phenotypes)}")
        
    return "\n".join(context_lines) if context_lines else "  No significant phenotype-to-disease associations identified."

def parse_variant_identifiers(variant_record: Dict[str, Any]) -> List[List[str]]:
    """
    Parses transcript and variant details from HGVS notation.
    """
    parsed_variants = []
    hgvs_notation = variant_record.get("AAChange.refGene", "") or variant_record.get("hgvs.c", "")
    
    if hgvs_notation:
        try:
            for segment in hgvs_notation.split(','):
                if ':' in segment:
                    components = segment.split(':')
                    if len(components) >= 2:
                        transcript_id = components[0].split('.')[0]
                        variant_detail = components[1]
                        parsed_variants.append([transcript_id, variant_detail])
        except Exception:
            pass
    return parsed_variants

def resolve_variant_nodes(query_parameters: List[List[str]], limit: int = 2) -> List[str]:
    """
    Queries the knowledge graph to resolve exact Variant node names based on parsed IDs.
    """
    resolved_names = []
    if not query_parameters: 
        return resolved_names
        
    try:
        with db_driver.session() as session:
            for transcript, variation in query_parameters:
                cypher_query = """
                MATCH (v:Variant)
                WHERE v.variant_name CONTAINS $transcript 
                  AND v.variant_name CONTAINS $variation
                RETURN v.variant_name AS exact_name LIMIT $limit
                """
                records = session.run(
                    cypher_query, 
                    transcript=transcript, 
                    variation=variation, 
                    limit=limit
                )
                for record in records:
                    if record.get("exact_name"):
                        resolved_names.append(record.get("exact_name"))
    except Exception:
        pass
        
    return list(set(resolved_names))

def retrieve_genomic_neighborhood(variant_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Executes a 3-hop Cypher traversal centered around Variant and Gene nodes.
    """
    parsed_queries = parse_variant_identifiers(variant_data)
    confirmed_variant_names = resolve_variant_nodes(parsed_queries)
    target_gene = variant_data.get("Gene.refGene", "")
    
    cypher_query = """ 
    UNWIND (CASE WHEN size($variants) > 0 THEN $variants ELSE [''] END) AS variant_name
    MATCH path1 = (v:Variant)-[*1..3]-(n1) 
    WHERE v.variant_name = variant_name 
    WITH path1 LIMIT $db_limit
    WITH { 
        nodes: [n IN nodes(path1) | { labels: labels(n), properties: apoc.map.removeKeys(properties(n), ['embedding']) }] 
    } AS subgraph
    RETURN subgraph
    
    UNION ALL
    
    UNWIND (CASE WHEN size($genes) > 0 THEN $genes ELSE [''] END) AS gene_name
    MATCH path2 = (g:Gene)-[*1..3]-(n2) 
    WHERE g.gene = gene_name 
    WITH path2 LIMIT $db_limit
    WITH { 
        nodes: [n IN nodes(path2) | { labels: labels(n), properties: apoc.map.removeKeys(properties(n), ['embedding']) }] 
    } AS subgraph
    RETURN subgraph
    """
    try:
        with db_driver.session() as session:
            result = session.run(
                cypher_query, 
                variants=confirmed_variant_names, 
                genes=[target_gene] if target_gene else [], 
                db_limit=CYPHER_PATH_LIMIT
            )
            return [record.get("subgraph") for record in result if record.get("subgraph")]
    except Exception as error:
        logger.error(f"Genomic KG Retrieval Exception: {error}")
        return []

def format_genomic_context(subgraphs: List[Dict[str, Any]]) -> str:
    """
    Structures the variant/gene subgraphs into a textual prompt context.
    """
    associated_diseases = []
    
    for subgraph in subgraphs:
        for node in subgraph.get("nodes", []):
            if "Disease" in node.get("labels", []):
                associated_diseases.append(node.get("properties", {}).get("disease_name", "Unknown"))
                
    unique_diseases = list(set(associated_diseases))[:30]
    if not unique_diseases:
        return "  No significant variant/gene-to-disease associations identified."
        
    return "\n".join([f"  - {disease}" for disease in unique_diseases])


# ==============================================================================
# 4. Core Inference Pipeline
# ==============================================================================

if __name__ == "__main__":
    
    logger.info("Initializing LLM Engine via vLLM...")
    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    inference_engine = LLM(model=MODEL_PATH, trust_remote_code=True)
    generation_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=8192)

    processed_samples = set()
    if os.path.exists(OUTPUT_DATA_PATH):
        with open(OUTPUT_DATA_PATH, "r", encoding="utf-8") as output_file:
            for completed_line in output_file:
                processed_samples.add(json.loads(completed_line).get("sample_index"))

    total_records = sum(1 for _ in open(INPUT_DATA_PATH, "r", encoding="utf-8"))

    with open(INPUT_DATA_PATH, "r", encoding="utf-8") as file_in, \
         open(OUTPUT_DATA_PATH, "a", encoding="utf-8") as file_out:

        progress_bar = tqdm(total=total_records, desc="GraphRAG Inference Pipeline", unit="sample")

        for jsonl_line in file_in:
            record_data = json.loads(jsonl_line.strip())
            current_index = record_data.get("sample_index")

            if current_index in processed_samples:
                progress_bar.update(1)
                continue

            # ------------------------------------------------------------------
            # Phase 1: Extract Base EHR & Genomic Data
            # ------------------------------------------------------------------
            ehr_content = record_data.get("trans_result", "").strip()
            genomic_variants = record_data.get("variants", [])[:3]

            if not ehr_content:
                progress_bar.update(1)
                continue

            # ------------------------------------------------------------------
            # Phase 2: Execute Knowledge Graph Retrieval
            # ------------------------------------------------------------------
            hpo_identifiers = extract_hpo_identifiers(record_data)
            raw_phenotype_graphs = retrieve_phenotype_neighborhood(hpo_identifiers)
            prioritized_pheno_graphs = prioritize_subgraphs_by_ic(raw_phenotype_graphs)
            phenotype_kg_context = format_phenotype_context(prioritized_pheno_graphs)

            variant_descriptions = []
            variant_kg_contexts = []
            
            for variant in genomic_variants:
                desc = (f"Gene {variant.get('Gene.refGene')}, location {variant.get('Chr')}:{variant.get('Start')}-{variant.get('End')}, "
                        f"nucleotide change {variant.get('Ref')}>{variant.get('Alt')}, protein change {variant.get('AAChange.refGene')}.")
                variant_descriptions.append(desc)
                
                raw_variant_graphs = retrieve_genomic_neighborhood(variant)[:MAX_SUBGRAPHS_LIMIT]
                formatted_var_context = format_genomic_context(raw_variant_graphs)
                variant_kg_contexts.append(f"Variant Target [{variant.get('Gene.refGene')}]:\n{formatted_var_context}")

            variant_content = "\n".join(variant_descriptions) if variant_descriptions else "No candidate genetic variants identified."
            knowledge_graph_content = (
                "--- Phenotype-Driven Subgraph Context ---\n"
                f"{phenotype_kg_context}\n\n"
                "--- Variant/Gene-Driven Subgraph Context ---\n"
                f"{chr(10).join(variant_kg_contexts) if variant_kg_contexts else 'No associations identified.'}"
            )

            # ------------------------------------------------------------------
            # Phase 3: Construct Diagnostic Prompt
            # ------------------------------------------------------------------
            final_prompt = (
                "You are a rare disease diagnostic expert.\n\n"
                "Patient Information:\n"
                f"{ehr_content}\n"
                f"{variant_content}\n\n"
                "Knowledge Graph Retrieval Results:\n"
                f"{knowledge_graph_content}\n\n"
                "Task:\n"
                "- Analyze the patient information and the retrieved knowledge graph data.\n"
                "- Enumerate the top 20 most likely rare disease diagnoses.\n\n"
                "Output Requirements:\n"
                "- List exactly 20 diagnoses.\n"
                "- Rank them in descending order of likelihood.\n"
                "- List one diagnosis per line.\n"
                "- Cover diverse possibilities (at least 20 unique diagnoses).\n"
                "- Do not provide any explanations or reasoning.\n\n"
                "Top 20 Diagnoses:"
            )

            # Applying ChatML template wrapping if required by the model architecture
            messages = [{"role": "user", "content": final_prompt}]
            tokenized_prompt = llm_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # ------------------------------------------------------------------
            # Phase 4: Generate Prediction & Persist
            # ------------------------------------------------------------------
            try:
                inference_output = inference_engine.generate([tokenized_prompt], generation_params, use_tqdm=False)
                diagnostic_prediction = inference_output[0].outputs[0].text.strip()

                file_out.write(json.dumps({
                    "sample_index": current_index,
                    "sampleID": record_data.get("sampleID"),
                    "diagnosis_results": diagnostic_prediction,
                }, ensure_ascii=False) + "\n")
                file_out.flush()

            except Exception as generation_error:
                logger.error(f"Inference Engine Error on Sample {current_index}: {generation_error}")

            progress_bar.update(1)

        progress_bar.close()
    
    db_driver.close()
    logger.info("RareMedRAG Pipeline Inference Successfully Terminated.")