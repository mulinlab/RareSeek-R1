"""Local end-to-end workflow. No external clinical-text API calls."""
import importlib.metadata
import platform
import re
from pathlib import Path
from .io import by_id, digest, load_config, load_json, read_jsonl, write_json, write_jsonl
from .graph import validate_graph
from .evaluation import evaluate
from .rag_support import patient_variants

def parse_diagnoses(text):
    if not isinstance(text, str):
        raise ValueError('Prediction text must be a string')
    if '<think>' in text and '</think>' not in text:
        return []
    text = text.rsplit('</think>', 1)[-1].strip()
    numbered, bullets = {}, []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r'^(?:\*\*)?(\d+)[.)]\s*(.*?)\s*$', line)
        if match:
            rank = int(match.group(1))
            if rank < 1 or rank > 1000 or rank in numbered:
                raise ValueError('Diagnosis ranks must be unique positive integers up to 1000')
            numbered[rank] = match.group(2).strip().strip('*').strip()
        else:
            match = re.match(r'^[-*]\s+(.+?)\s*$', line)
            if match:
                bullets.append(match.group(1).strip().strip('*').strip())
    # Missing or repeated disease names retain their original rank positions.
    return [numbered.get(i, '') for i in range(1, max(numbered) + 1)] if numbered else bullets


def normalize_names(names, graph):
    mapping = {}
    for disease in graph["diseases"]:
        if not disease['id'].startswith(('ORPHA:', 'OMIM:')):
            continue
        for name in [disease["name"], disease["id"], *disease.get("aliases", [])]:
            mapping.setdefault(name.casefold().strip(), set()).add(disease["id"])
    # Ambiguous names are unmapped rather than credited with multiple diseases.
    return [sorted(mapping.get(n.casefold().strip(), [])) if len(mapping.get(n.casefold().strip(), [])) == 1 else [] for n in names]

def prepare_records(rows, config):
    by_id(rows)
    translations = load_json(config["translations"]) if config.get("translations") else {}
    output = []
    for row in rows:
        # Explicit allowlist prevents gold labels or existing predictions entering inference.
        text = row.get("context")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{row['sample_id']}: nonempty context required")
        language = row.get("language", "en")
        if language != "en":
            text = translations.get(row["sample_id"])
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{row['sample_id']}: provide a local English translation; clinical text is not sent to an API")
        hpos = row.get("hpo_list", [])
        if not isinstance(hpos, list) or any(not isinstance(h, str) for h in hpos):
            raise ValueError("hpo_list must be a list of strings")
        variants = patient_variants(row)
        if not isinstance(variants, list) or any(not isinstance(v, dict) for v in variants):
            raise ValueError("variants must be a list of objects")
        output.append({"sample_id": row["sample_id"], "id": row.get("id", row["sample_id"]),
                       "context": text, "hpo_list": sorted(set(hpos)), "variants": variants})
    return output

def preflight(config):
    if config.get("backend") != "vllm":
        raise ValueError("backend must be vllm")
    if config.get("mode") != "standard":
        raise ValueError("Use RareMedRAG.py for graph-augmented inference")
    for key in ("input", "gold", "graph", "hierarchy"):
        if (key == "input" or config.get(key)) and (not config.get(key) or not Path(config[key]).is_file()):
            raise FileNotFoundError(f"Missing {key}: {config.get(key)}")
    graph = validate_graph(load_json(config["graph"])) if config.get("graph") else None
    rows = read_jsonl(config["input"])
    prepared = prepare_records(rows, config)
    gold = read_jsonl(config["gold"]) if config.get("gold") else None
    if gold is not None:
        if graph is None:
            raise ValueError("Evaluation requires --graph for disease-name normalization")
        if set(by_id(prepared)) != set(by_id(gold)):
            raise ValueError("Input and gold cohorts differ")
        evaluate(gold, [], load_json(config["hierarchy"]) if config.get("hierarchy") else {})
    if type(config.get("top_k", 20)) is not int or config.get("top_k", 20) < 1:
        raise ValueError("top_k must be a positive integer")
    model = config.get('model', {})
    if any(type(model.get(k, default)) is not int or model.get(k, default) < 1 for k, default in
           [('tensor_parallel_size', 1), ('max_model_len', 8192), ('max_tokens', 2048)]):
        raise ValueError('GPU and token limits must be positive integers')
    if model.get('max_tokens', 2048) >= model.get('max_model_len', 8192):
        raise ValueError('max_model_len must exceed max_tokens')
    if config["backend"] == "vllm":
        model = config.get("model", {})
        if not model.get("path"):
            raise ValueError("A model path is required")
        if Path(model['path']).is_absolute() and not Path(model['path']).is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {model['path']}")
        if not Path(model["path"]).is_dir() and not model.get("revision"):
            raise ValueError("Provide an existing local model directory, or a model ID with --revision")
    return prepared, gold, graph

def run(config):
    records, gold, graph = preflight(config)
    out = Path(config["output_dir"])
    if out.exists() and any(out.iterdir()) and not config.get("overwrite", False):
        raise FileExistsError("Output already contains a run; choose a new output_dir or explicitly set overwrite=true")
    out.mkdir(parents=True, exist_ok=True)
    top_k = config.get("top_k", 20)
    prompts = []
    for record in records:
        prompts.append(f"Patient information:\n{record['context']}\nVariants: {record['variants']}\n"
                       f"List up to {top_k} most likely diagnoses, one disease name per numbered line, ranked by likelihood. "
                       "Do not include explanations. If insufficient information, return an empty answer.")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    model = config["model"]
    kwargs = {} if Path(model["path"]).is_dir() else {"revision": model["revision"]}
    tokenizer = AutoTokenizer.from_pretrained(model["path"], **kwargs)
    engine = LLM(model=model["path"], tensor_parallel_size=model.get("tensor_parallel_size", 1),
                 max_model_len=model.get("max_model_len", 8192), seed=config.get("seed", 42), **kwargs)
    templated = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    budget = model.get("max_tokens", 2048)
    if any(len(tokenizer.encode(p, add_special_tokens=False)) + budget > model.get("max_model_len", 8192) for p in templated):
        raise ValueError("Input + output token budget exceeds model.max_model_len")
    sampling = SamplingParams(temperature=model.get("temperature", 0), top_p=model.get("top_p", 1),
                              max_tokens=budget, seed=config.get("seed", 42))
    token_ids = [tokenizer.encode(p, add_special_tokens=False) for p in templated]
    texts = [o.outputs[0].text for o in engine.generate([{'prompt_token_ids': ids} for ids in token_ids], sampling_params=sampling)]
    if len(texts) != len(records):
        raise RuntimeError('Inference returned a different number of results than inputs')
    predictions = []
    for record, text in zip(records, texts):
        names = parse_diagnoses(text)[:top_k]
        prediction = {"sample_id": record["sample_id"], "id": record["id"], "predict_result": text,
                      "extracted_diseases": names, "status": "ok" if names else "empty",
                      "backend": config["backend"], "data_kind": config.get("data_kind", "private")}
        if graph is not None:
            codes = normalize_names(names, graph)
            prediction.update(predict_code=codes, status="ok" if any(codes) else "unmapped_or_empty")
        predictions.append(prediction)
    report = evaluate(gold, predictions, load_json(config["hierarchy"]) if config.get("hierarchy") else {}) if gold is not None else {"total": len(predictions)}
    report.update({"data_kind": config.get("data_kind", "private"), "backend": config["backend"],
                   "retriever": "none"})
    write_jsonl(out / "prepared.jsonl", records)
    write_jsonl(out / "prompts.jsonl", [{"sample_id": r["sample_id"], "prompt": p} for r, p in zip(records, prompts)])
    write_jsonl(out / "predictions.jsonl", predictions)
    if gold is not None:
        write_json(out / "metrics.json", report)
    elif config.get('overwrite') and (out / 'metrics.json').exists():
        (out / 'metrics.json').unlink()
    versions = {}
    for package in ("vllm", "torch", "transformers", "neo4j"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(out / "manifest.json", {"config": config, "python": platform.python_version(),
        "platform": platform.platform(), "packages": versions,
        "inputs_sha256": {k: digest(config[k]) for k in ("input", "gold", "graph", "hierarchy", "translations") if config.get(k)},
        "source_sha256": {str(p.relative_to(Path(__file__).parent)): digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "outputs_sha256": {p.name: digest(p) for p in sorted(out.glob("*")) if p.is_file() and p.name != "manifest.json"}})
    return report
