"""Run standard inference from clinical JSONL and local model weights."""
import argparse
from pathlib import Path
from rareseek.io import ROOT, path, load_config
from rareseek.pipeline import run
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="sample_data/sample.jsonl")
    p.add_argument("--model", default="models/RareSeek-R1")
    p.add_argument("--output", default="outputs/standard", help="Output directory")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--revision", help="Fixed revision when loading a remote model ID")
    p.add_argument("--graph", help="Optional disease graph for name normalization")
    p.add_argument("--gold", help="Optional reference labels for evaluation")
    p.add_argument("--hierarchy", help="Optional hierarchy for evaluation")
    p.add_argument("--config", help="Optional complete configuration for batch experiments")
    args = p.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        model_path = path(args.model)
        is_local = model_path.is_dir() or args.model.startswith((".", "/", "models/")) or Path(args.model).is_absolute()
        if is_local and not model_path.is_dir():
            p.error(f"Model directory does not exist: {model_path}")
        config = {
            "input": str(path(args.input)), "output_dir": str(path(args.output)),
            "backend": "vllm", "data_kind": "private", "top_k": 20, "seed": 42,
            "model": {"path": str(model_path) if is_local else args.model,
                      "revision": args.revision, "tensor_parallel_size": args.tensor_parallel_size,
                      "max_model_len": args.max_model_len, "max_tokens": args.max_tokens,
                      "temperature": 0.1, "top_p": 0.95},
        }
        for key in ("graph", "gold", "hierarchy"):
            if getattr(args, key):
                config[key] = str(path(getattr(args, key)))
    config["mode"] = "standard"
    run(config)
