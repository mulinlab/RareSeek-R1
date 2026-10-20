import argparse
import json
from .io import ROOT, load_config, load_json, path, read_jsonl, write_json
from .pipeline import preflight, run
from .evaluation import evaluate

def main():
    parser = argparse.ArgumentParser(description="RareSeek inference and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "preflight"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--gold", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--hierarchy", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "evaluate":
            result = evaluate(read_jsonl(path(args.gold)), read_jsonl(path(args.predictions), allow_empty=True), load_json(path(args.hierarchy)))
            write_json(path(args.output), result)
        else:
            config = load_config(args.config)
            if args.command == "run":
                result = run(config)
            elif args.command == "preflight":
                records, gold, graph = preflight(config)
                result = {"ok": True, "cases": len(records), "diseases": len(graph["diseases"]) if graph else 0, "backend": config["backend"]}
        print(json.dumps({k: v for k, v in result.items() if k != "per_sample"}, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")

if __name__ == "__main__":
    main()
