"""Strict UTF-8 input/output and paths independent of the working directory."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def path(value, base=ROOT):
    result = Path(os.path.expandvars(str(value))).expanduser()
    return result.resolve() if result.is_absolute() else (base / result).resolve()

def model_location(value, base=ROOT):
    candidate = path(value, base)
    if candidate.is_dir() or Path(value).is_absolute() or str(value).startswith(('.', '/', 'models/', 'models\\')):
        return str(candidate)
    return str(value)


def load_json(filename):
    with Path(filename).open(encoding="utf-8-sig") as stream:
        return json.load(stream)

def record_id(row):
    """Accept original numeric/string id and the internal sample_id alias."""
    identifiers = [row[k] for k in ("sample_id", "id") if k in row]
    if not identifiers or any(type(v) not in (str, int) or not str(v).strip() for v in identifiers):
        raise ValueError("Each record requires a nonempty id or sample_id")
    normalized = [str(v).strip() for v in identifiers]
    if len(set(normalized)) != 1:
        raise ValueError("id and sample_id refer to different records")
    return normalized[0]

def read_jsonl(filename, allow_empty=False):
    rows = []
    with Path(filename).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{filename}:{number}: expected an object")
                value["sample_id"] = record_id(value)
                rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"Empty dataset: {filename}")
    return rows

def write_json(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(filename)

def write_jsonl(filename, rows):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(filename)

def by_id(rows):
    result = {}
    for row in rows:
        sid = record_id(row)
        if not isinstance(sid, str) or not sid.strip() or sid in result:
            raise ValueError(f"Missing, invalid or duplicate sample_id: {sid!r}")
        result[sid] = row
    return result

def digest(filename):
    h = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_config(filename):
    filename = path(filename)
    config = load_json(filename)
    for key in ("input", "gold", "graph", "hierarchy", "output_dir", "translations"):
        if config.get(key):
            config[key] = str(path(config[key], filename.parent))
    config["config_file"] = str(filename)
    model = config.get("model", {})
    if model.get('path'):
        model['path'] = model_location(model['path'], filename.parent)
    return config
