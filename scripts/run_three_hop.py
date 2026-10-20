"""Build, import, run three-hop RAG, normalize, and evaluate locally.

Requires prepared licensed/public source files, a fresh Neo4j database and local
model weights. Does not download or publish clinical data.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rareseek.io import ROOT, path, read_jsonl, by_id, load_json, write_json, write_jsonl, digest
from rareseek.pipeline import parse_diagnoses, normalize_names
from rareseek.evaluation import evaluate
from rareseek.rag_support import positive_hpos, patient_variants


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('raw', 'processed', 'cases', 'gold', 'output', 'model', 'revision'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--database', default=os.environ.get('NEO4J_DATABASE', 'neo4j'))
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--path-limit', type=int, default=2000)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--max-tokens', type=int, default=8192)
    a = p.parse_args()
    raw, processed, cases, gold, output = (path(getattr(a, k)) for k in ('raw','processed','cases','gold','output'))
    if output.exists():
        p.error('Use a new output directory for each experiment')
    model = path(a.model)
    if not model.is_dir():
        p.error(f'Model directory does not exist: {model}')
    if not os.environ.get('NEO4J_PASSWORD'):
        p.error('NEO4J_PASSWORD is required')
    case_rows, gold_rows = read_jsonl(cases), read_jsonl(gold)
    for case in case_rows:
        positive_hpos(case)
        patient_variants(case)
        if not isinstance(case.get('context'), str) or not case['context'].strip():
            p.error('Each case requires nonempty context')
    evaluate(gold_rows, [])
    if set(by_id(case_rows)) != set(by_id(gold_rows)):
        p.error('Cases and gold must have identical sample IDs')
    output.mkdir(parents=True)
    commands = []
    def run(script, *args):
        command = [sys.executable, str(ROOT / script), *map(str, args)]
        subprocess.run(command, cwd=ROOT, check=True)
        commands.append([script, *map(str, args)])
    run('scripts/build_kg.py', '--raw', raw, '--processed', processed)
    run('scripts/graph_snapshot.py', '--directory', processed / 'neo4j_import', '--output', output / 'snapshot', '--data-kind', 'private')
    run('scripts/import_neo4j.py', '--directory', processed / 'neo4j_import', '--database', a.database)
    run('RareMedRAG.py', '--input', cases, '--output', output / 'raw_predictions.jsonl',
        '--model', model, '--revision', a.revision, '--database', a.database,
        '--ic-source', output / 'snapshot/HPO_phenotype.json', '--tensor-parallel-size', a.tensor_parallel_size,
        '--path-limit', a.path_limit, '--max-model-len', a.max_model_len, '--max-tokens', a.max_tokens)
    graph = load_json(output / 'snapshot/graph.json')
    predictions = read_jsonl(output / 'raw_predictions.jsonl')
    for row in predictions:
        row['extracted_diseases'] = parse_diagnoses(row['predict_result'])
        row['predict_code'] = normalize_names(row['extracted_diseases'], graph)
    write_jsonl(output / 'predictions.jsonl', predictions)
    metrics = evaluate(gold_rows, predictions, load_json(output / 'snapshot/hierarchy.json'))
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'workflow_manifest.json', {
        'commands': commands, 'ic': '-ln(n/N); no smoothing; no ancestor propagation',
        'cases_sha256': digest(cases), 'gold_sha256': digest(gold),
        'graph_csv_sha256': {f.name: digest(f) for f in sorted((processed / 'neo4j_import').glob('*.csv'))},
        'note': 'Compare metrics to the paper only after matching cohorts, source snapshots, weights and protocol.'})


if __name__ == '__main__':
    main()
