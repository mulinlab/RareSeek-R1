"""Shared chromosome, coordinate and transcript-variant conventions."""
import json
import re

MISSING = {"", ".", "-", "none", "null", "nan", "n/a", "unknown"}


def text(value):
    result = "" if value is None else str(value).strip()
    return "" if result.casefold() in MISSING else result


def chromosome(value):
    value = re.sub(r"^chr", "", text(value), flags=re.I).upper()
    return "MT" if value == "M" else value


def coordinates(record):
    def pick(*names):
        return next((text(record.get(k)) for k in names if text(record.get(k))), "")
    chrom = chromosome(pick("chr", "chrom", "Chr", "Chromosome"))
    pos, ref, alt = pick("pos", "Start", "PositionVCF"), pick("ref", "Ref"), pick("alt", "Alt")
    if not all((chrom, pos, ref, alt)):
        return None
    if not re.fullmatch(r"(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)", chrom) or not pos.isdigit() or int(pos) < 1:
        raise ValueError("Variant coordinates require a chromosome and positive integer position")
    if not re.fullmatch(r"[ACGTNacgtn]+", ref) or not re.fullmatch(r"[ACGTNacgtn]+", alt):
        raise ValueError("Variant coordinate lookup requires explicit nucleotide alleles")
    return {"chr": chrom, "pos": str(int(pos)), "ref": ref.upper(), "alt": alt.upper()}


def transcript_changes(record):
    notation = text(record.get("AAChange.refGene")) or text(record.get("hgvs.c"))
    if not notation:
        return []
    found = set()
    for segment in notation.split(","):
        components = [part.strip() for part in segment.split(":")]
        transcript = next((p for p in components if re.fullmatch(r"(?:NM_|NR_)\d+(?:\.\d+)?|ENST\d+(?:\.\d+)?", p)), None)
        change = next((p for p in components if re.fullmatch(r"[cn]\.\S+", p)), None)
        if not transcript or not change:
            raise ValueError("Transcript HGVS requires NM_/NR_/ENST accession and c. or n. change")
        found.add((transcript, change))
    return [list(v) for v in sorted(found)]


def variant_name(record):
    transcript, change = text(record.get("consequence_transcript")), text(record.get("consequence_hgvs_c"))
    if transcript and change:
        # Some annotation sources include the accession in hgvs_c itself.
        if ":" in change:
            accession, change = change.rsplit(":", 1)
            if accession != transcript:
                raise ValueError("Consequence transcript disagrees with hgvs_c accession")
        protein = text(record.get("consequence_hgvs_p"))
        return ":".join([transcript, change] + ([protein] if protein else []))
    coord = coordinates(record)
    if coord:
        return f"variant:{coord['chr']}:{coord['pos']}:{coord['ref']}>{coord['alt']}"
    raise ValueError("Variant requires a transcript/change or complete genomic coordinates")


def merge_annotations(previous, current):
    """Preserve source records; never select one conflicting numeric annotation."""
    records = json.loads(previous.get("annotations", "[]")) if previous else []
    current = {k: text(v) for k, v in current.items() if k != "annotations"}
    records.append(current)
    records = [json.loads(v) for v in sorted({json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records})]
    merged = {}
    for key in set().union(*(r.keys() for r in records)):
        values = {r[key] for r in records if r.get(key)}
        if key in {"chr", "pos", "ref", "alt"} and len(values) > 1:
            raise ValueError(f"Same variant identifier has conflicting {key} values: {sorted(values)}")
        if key == "Source":
            merged[key] = "|".join(sorted(values))
        else:
            merged[key] = next(iter(values)) if len(values) == 1 else ""
    merged["annotations"] = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return merged


def annotation_frame(frame, expected=None):
    """Validate provider coordinate coverage and serialize nested TSV cells as JSON."""
    if not {'chr', 'pos', 'ref', 'alt'}.issubset(frame.columns):
        raise ValueError('Annotation response is missing chr/pos/ref/alt')
    frame = frame.copy()
    normalized = [coordinates(row) for row in frame.to_dict('records')]
    if any(row is None for row in normalized):
        raise ValueError('Annotation response contains unresolved coordinates')
    if expected is not None:
        expected_keys = {tuple(coordinates(row).values()) for row in expected.to_dict('records')}
        returned_keys = {tuple(row.values()) for row in normalized}
        if expected_keys != returned_keys:
            raise ValueError('Annotation response does not cover exactly the requested variants')
    for key in ('chr', 'pos', 'ref', 'alt'):
        frame[key] = [row[key] for row in normalized]
    for key in frame:
        frame[key] = frame[key].map(lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False)
                                   if isinstance(value, (dict, list)) else value)
    return frame
