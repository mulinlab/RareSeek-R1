"""
RareMedRAG Pipeline for Rare Disease Diagnosis
"""

import os
import json
import math
import re
import logging
from typing import Any, Dict, List, Tuple

from rareseek.io import ROOT, path, read_jsonl, by_id, write_json, digest, model_location, load_json
from pathlib import Path
import argparse
from rareseek.ic import information_content_records, mean_information_content
from rareseek.variants import transcript_changes, coordinates, text as variant_text
from rareseek.rag_support import subgraph_key, unique_subgraphs, checked_paths, variant_matches, positive_hpos, patient_variants

# ==============================================================================
# 1. System Configuration & Initialization
# ==============================================================================



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model Configuration
MODEL_PATH = "TaoMedAI/RareSeek-R1"

# Neo4j Database Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Graph Retrieval Constraints
HPO_IC_FILE_PATH = os.getenv("HPO_IC_PATH", "HPO_phenotype.json")
MAX_SUBGRAPHS_LIMIT = 300
CYPHER_PATH_LIMIT = 2000

# Input/Output Paths
INPUT_DATA_PATH = None
OUTPUT_DATA_PATH = None

db_driver = None

# ==============================================================================
# 2. Utility & Pre-computation Functions
# ==============================================================================

def compute_information_content(file_path: str) -> Dict[str, float]:
    """
    Computes Information Content (IC) for Human Phenotype Ontology (HPO) terms.
    """
    return information_content_records(load_json(path(file_path)))


def load_hpo_terminology(file_path: str = "zh-HPO.json") -> Dict[str, str]:
    """
    Loads mapping of HPO IDs to human-readable terms.
    """
    data = load_json(path(file_path))
    if not isinstance(data, dict) or any(not isinstance(v, dict) for v in data.values()):
        raise ValueError('HPO terminology must be keyed by HPO ID')
    return {k: v.get('hpo_term') or k for k, v in data.items()}

# Global Initializations
HPO_IC_SCORES = {}
HPO_TERMINOLOGY = {}

# ==============================================================================
# 3. Knowledge Graph Retrieval Modules
# ==============================================================================

def extract_hpo_identifiers(record: Dict[str, Any]) -> List[str]:
    """
    Extracts HPO IDs from various possible fields within a clinical record.
    """
    return positive_hpos(record)


def retrieve_phenotype_neighborhood(phenotype_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Executes a 3-hop Cypher traversal centered around provided Phenotype nodes.
    """
    if not phenotype_ids:
        return []
        
    cypher_query = """
    UNWIND $phenotypes AS phenotype_id
    MATCH path = (p:Phenotype {HPOId: phenotype_id})-[*1..3]-(related_node)
    WHERE all(n IN nodes(path) WHERE single(m IN nodes(path) WHERE m = n))
    WITH DISTINCT path ORDER BY [n IN nodes(path) | coalesce(n.ORPHAcode,n.HPOId,n.Symbol,n.variant_name)], [r IN relationships(path) | type(r)] LIMIT $db_limit
    WITH {
        nodes: [n IN nodes(path) | { 
            labels: labels(n), 
            properties: properties(n) 
        }],
        relationships: [r IN relationships(path) | { type: type(r), source: properties(startNode(r)), target: properties(endNode(r)), properties: properties(r) }]
    } AS subgraph
    RETURN subgraph
    """
    try:
        with db_driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                cypher_query, 
                phenotypes=phenotype_ids, 
                db_limit=CYPHER_PATH_LIMIT + 1
            )
            return checked_paths([record.get("subgraph") for record in result if record.get("subgraph")], CYPHER_PATH_LIMIT)
    except Exception as error:
        raise RuntimeError("Phenotype KG retrieval failed") from error

def prioritize_subgraphs_by_ic(subgraphs: List[Dict[str, Any]], limit=MAX_SUBGRAPHS_LIMIT, report=None) -> List[Dict[str, Any]]:
    """
    Ranks extracted subgraphs based on the average Information Content of their Phenotypes.
    """
    scored_subgraphs = []
    
    for subgraph in unique_subgraphs(subgraphs):
        unique_phenotypes = set()
        for node in subgraph.get("nodes", []):
            if "Phenotype" in node.get("labels", []):
                properties = node.get("properties", {})
                hpo_id = properties.get("HPOId") or properties.get("hpo_id")
                if hpo_id:
                    unique_phenotypes.add(str(hpo_id).strip())
        
        missing = sorted(unique_phenotypes - HPO_IC_SCORES.keys())
        if missing:
            if report is not None:
                report['unscored_paths'] = report.get('unscored_paths', 0) + 1
                report.setdefault('unannotated_hpo_ids', set()).update(missing)
            continue
        average_ic = mean_information_content(unique_phenotypes, HPO_IC_SCORES)

        subgraph_copy = dict(subgraph)
        subgraph_copy["_average_ic"] = average_ic
        scored_subgraphs.append(subgraph_copy)
        
    scored_subgraphs.sort(key=lambda x: (-x["_average_ic"], subgraph_key(x)))
    return scored_subgraphs if limit is None else scored_subgraphs[:limit]

def parse_variant_identifiers(variant_record):
    return transcript_changes(variant_record)


def resolve_variant_nodes(query_parameters: List[List[str]], limit: int = 1000) -> List[str]:
    """
    Queries the knowledge graph to resolve exact Variant node names based on parsed IDs.
    """
    resolved_names = []
    if not query_parameters: 
        return resolved_names
        
    try:
        with db_driver.session(database=NEO4J_DATABASE) as session:
            for transcript, variation in query_parameters:
                cypher_query = """
                MATCH (v:Variant)
                WHERE any(token IN split(v.variant_name, ':') WHERE token = $transcript OR ($unversioned AND token =~ ($transcript + '[.][0-9]+')))
                  AND $variation IN split(v.variant_name, ':')
                RETURN v.variant_name AS exact_name ORDER BY exact_name LIMIT $limit
                """
                records = session.run(
                    cypher_query, 
                    transcript=transcript, 
                    variation=variation, 
                    limit=limit + 1,
                    unversioned="." not in transcript
                )
                records = list(records)
                if len(records) > limit:
                    raise RuntimeError("Variant resolution cap exceeded")
                for record in records:
                    name = record.get("exact_name")
                    if name and variant_matches(name, transcript, variation):
                        resolved_names.append(name)
    except Exception as error:
        raise RuntimeError("Variant lookup failed") from error

    return sorted(set(resolved_names))

def retrieve_seed_neighborhood(label, key, identifiers):
    if (label, key) not in {("Variant", "variant_name"), ("Gene", "Symbol")}:
        raise ValueError("Unsupported seed type")
    if not identifiers:
        return []
    query = f"""
    UNWIND $identifiers AS identifier
    MATCH path = (seed:{label} {{{key}: identifier}})-[*1..3]-(related)
    WHERE all(n IN nodes(path) WHERE single(m IN nodes(path) WHERE m = n))
    WITH DISTINCT path
    ORDER BY [n IN nodes(path) | coalesce(n.ORPHAcode,n.HPOId,n.Symbol,n.variant_name)],
             [r IN relationships(path) | type(r)]
    LIMIT $limit
    RETURN {{nodes: [n IN nodes(path) | {{labels: labels(n), properties: properties(n)}}],
            relationships: [r IN relationships(path) | {{type: type(r),
              source: properties(startNode(r)), target: properties(endNode(r)), properties: properties(r)}}]}} AS subgraph
    """
    with db_driver.session(database=NEO4J_DATABASE) as session:
        rows = [r["subgraph"] for r in session.run(query, identifiers=identifiers, limit=CYPHER_PATH_LIMIT + 1)]
    return checked_paths(rows, CYPHER_PATH_LIMIT)


def retrieve_variant_neighborhood(variant):
    names = resolve_variant_nodes(parse_variant_identifiers(variant))
    # Genomic lookup is an exact-coordinate fallback, not fuzzy allele matching.
    if not names:
        coord = coordinates(variant)
        if coord:
            with db_driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(
                    "MATCH (v:Variant {chr:$chr, pos:$pos, ref:$ref, alt:$alt}) "
                    "RETURN v.variant_name AS name ORDER BY name LIMIT $limit",
                    **coord, limit=1001)
                names = [r['name'] for r in result]
            if len(names) > 1000:
                raise RuntimeError('Coordinate lookup exceeds 1000 variant nodes')
    return retrieve_seed_neighborhood("Variant", "variant_name", names)


def retrieve_gene_neighborhood(gene):
    return retrieve_seed_neighborhood("Gene", "Symbol", [gene])


def build_retrieval_context(record):
    from rareseek.rag_support import allocate_evidence, render_evidence
    variants = record.get("variants", [])
    if not isinstance(variants, list) or any(not isinstance(v, dict) for v in variants):
        raise ValueError("variants must be a list of objects")
    variant_groups = []
    genes = set()
    for index, variant in enumerate(variants, 1):
        gene_text = variant_text(variant.get("Gene.refGene"))
        if not isinstance(variant.get('Gene.refGene', ''), (str, type(None))):
            raise ValueError('Gene.refGene must be a string')
        genes.update(g.strip() for g in re.split(r"[;,]", gene_text) if g.strip() and g.strip() != ".")
        paths = retrieve_variant_neighborhood(variant)
        variant_groups.append([{"kind": "variant", "target": f"variant_{index}",
                                "input": variant, "path": graph} for graph in paths] or
                              [{"kind": "variant", "target": f"variant_{index}", "input": variant,
                                "status": "no matching variant paths"}])
    gene_groups = []
    for gene in sorted(genes):
        paths = retrieve_gene_neighborhood(gene)
        gene_groups.append([{"kind": "gene", "target": gene, "path": graph} for graph in paths] or
                           [{"kind": "gene", "target": gene, "status": "no matching gene paths"}])
    ic_report = {}
    phenotype_paths = prioritize_subgraphs_by_ic(retrieve_phenotype_neighborhood(extract_hpo_identifiers(record)), limit=None, report=ic_report)
    phenotype_rows = [{"kind": "phenotype", "ic_score": graph["_average_ic"], "path": graph}
                      for graph in phenotype_paths]
    selected, report = allocate_evidence(variant_groups, gene_groups, phenotype_rows, 300)
    report['phenotype_ic_filter'] = {k: sorted(v) if isinstance(v, set) else v for k, v in ic_report.items()}
    return render_evidence(selected, HPO_TERMINOLOGY), report


# ==============================================================================
# 4. Core Inference Pipeline
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Three-hop RareMedRAG; requires a populated Neo4j and local vLLM")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--ic-source", required=True)
    parser.add_argument("--terminology", default=str(ROOT / "zh-HPO.json"))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--path-limit", type=int, default=2000)
    parser.add_argument("--database", default=NEO4J_DATABASE)
    args = parser.parse_args()
    if args.path_limit < 1 or args.tensor_parallel_size < 1 or args.max_tokens < 1 or args.max_model_len <= args.max_tokens:
        parser.error("Use positive retrieval/GPU/token limits and max-model-len greater than max-tokens")
    CYPHER_PATH_LIMIT = args.path_limit
    NEO4J_DATABASE = args.database
    INPUT_DATA_PATH, OUTPUT_DATA_PATH = str(path(args.input)), str(path(args.output))
    if Path(OUTPUT_DATA_PATH).exists():
        parser.error("Output exists; use a fresh path to avoid mixing experimental runs")
    records = read_jsonl(INPUT_DATA_PATH)
    by_id(records)
    for row in records:
        extract_hpo_identifiers(row)
        if not isinstance(row.get("context"), str) or not row["context"].strip():
            parser.error("Every input requires nonempty context and unique id/sample_id")
    if not NEO4J_PASSWORD:
        parser.error("NEO4J_PASSWORD is required")
    HPO_IC_SCORES = compute_information_content(str(path(args.ic_source)))
    HPO_TERMINOLOGY = load_hpo_terminology(str(path(args.terminology)))
    from neo4j import GraphDatabase
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    db_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    db_driver.verify_connectivity()
    with db_driver.session(database=NEO4J_DATABASE) as session:
        if not session.run("MATCH (d:Disease) RETURN count(d) AS n").single()["n"]:
            parser.error("Neo4j contains no Disease nodes")
    # IC must describe the same disease population and direct edges as the imported graph.
    ic_records = load_json(path(args.ic_source))
    with db_driver.session(database=NEO4J_DATABASE) as session:
        observed = {r["id"]: set(r["hpos"]) for r in session.run(
            "MATCH (d:Disease) OPTIONAL MATCH (d)-[:HAS_PHENOTYPE]->(p:Phenotype) "
            "RETURN d.ORPHAcode AS id, collect(DISTINCT p.HPOId) AS hpos")}
        expected = {d: {h["id"] if isinstance(h, dict) else h for h in (info.get("hpo_list", []) or [])} for d, info in ic_records.items()}
        if observed != expected:
            raise ValueError("IC source does not match database disease population/direct phenotype annotations")
    model_candidate = path(args.model)
    MODEL_PATH = model_location(args.model)
    if Path(MODEL_PATH).is_absolute() and not model_candidate.is_dir():
        parser.error(f'Model directory does not exist: {model_candidate}')
    model_kwargs = {} if Path(MODEL_PATH).is_dir() else {"revision": args.revision}
    Path(OUTPUT_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)

    
    logger.info("Initializing LLM Engine via vLLM...")
    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, **model_kwargs)
    inference_engine = LLM(model=MODEL_PATH, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len, seed=42, **model_kwargs)
    generation_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=args.max_tokens, seed=42)

    total_records = len(records)
    with open(OUTPUT_DATA_PATH, "x", encoding="utf-8") as file_out:
        progress_bar = tqdm(total=total_records, desc="GraphRAG Inference Pipeline", unit="sample")
        for record_data in records:
            current_index = record_data["sample_id"]

            # ------------------------------------------------------------------
            # Phase 1: Extract Base EHR & Genomic Data
            # ------------------------------------------------------------------
            ehr_content = record_data["context"].strip()
            knowledge_graph_content, retrieval_report = build_retrieval_context(record_data)

            # ------------------------------------------------------------------
            # Phase 3: Construct Diagnostic Prompt
            # ------------------------------------------------------------------
            final_prompt = (
                "You are a rare disease diagnostic expert.\n\n"
                "Patient Information:\n"
                f"{ehr_content}\n"
                f"Confirmed positive phenotypes: {[(h, HPO_TERMINOLOGY.get(h, h)) for h in extract_hpo_identifiers(record_data)]}\n"
                f"Patient candidate variants: {json.dumps(patient_variants(record_data), ensure_ascii=False)}\n\n"
                "Knowledge Graph Retrieval Results:\n"
                f"{knowledge_graph_content}\n\n"
                "Task:\n"
                "- Analyze the patient information and the retrieved knowledge graph data.\n"
                "- Enumerate the top 20 most likely rare disease diagnoses.\n\n"
                "Output Requirements:\n"
                "- List exactly 20 diagnoses.\n"
                "- Rank them in descending order of likelihood.\n"
                "- List one diagnosis per numbered line (1. Disease name).\n"
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
                prompt_ids = llm_tokenizer.encode(tokenized_prompt, add_special_tokens=False)
                if len(prompt_ids) + args.max_tokens > args.max_model_len:
                    raise ValueError("Prompt plus generation budget exceeds max-model-len")
                inference_output = inference_engine.generate([{'prompt_token_ids': prompt_ids}], sampling_params=generation_params, use_tqdm=False)
                diagnostic_prediction = inference_output[0].outputs[0].text.strip()

                file_out.write(json.dumps({
                    "sample_id": current_index,
                    "id": record_data.get("id", current_index),
                    "sampleID": record_data.get("sampleID"),
                    "predict_result": diagnostic_prediction,
                    "retriever": "three-hop",
                    "retrieval_context": knowledge_graph_content,
                    "retrieval_report": retrieval_report,
                }, ensure_ascii=False) + "\n")
                file_out.flush()

            except Exception as generation_error:
                raise RuntimeError(f"Inference failed for {current_index}") from generation_error

            progress_bar.update(1)

        progress_bar.close()
    
    write_json(OUTPUT_DATA_PATH + ".manifest.json", {
        "arguments": vars(args), "retriever": "three-hop", "seed": 42,
        "input_sha256": digest(INPUT_DATA_PATH), "ic_source_sha256": digest(path(args.ic_source)),
        "terminology_sha256": digest(path(args.terminology)), "output_sha256": digest(OUTPUT_DATA_PATH),
        "ic_policy": "all-records; natural-log; per-record deduplication; no ancestor propagation; no smoothing",
        "subgraph_score": "mean over all unique path phenotypes, including zero IC",
        "evidence_policy": "all variants queried; strict variant then gene then IC-descending phenotype priority; shared 300-path budget",
        "database_snapshot": "Record the imported CSV checksums separately"
    })
    db_driver.close()
    logger.info("RareMedRAG Pipeline Inference Successfully Terminated.")
