"""Information content: -ln(disease-record frequency / all records).

Each dictionary key counts once, including records without annotations. No HPO
ancestor propagation, patient weighting, smoothing or ontology-ID merging.
"""
import math
import re

def information_content_records(records):
    if not isinstance(records, dict) or not records:
        raise ValueError("IC source must be a nonempty disease-keyed dictionary")
    frequency = {}
    for disease, info in records.items():
        if not isinstance(disease, str) or not disease.strip() or not isinstance(info, dict):
            raise ValueError("IC disease IDs and records are malformed")
        terms = info.get("hpo_list", []) or []
        if not isinstance(terms, list):
            raise ValueError(f"{disease}: hpo_list must be a list")
        identifiers = set()
        for term in terms:
            identifier = term.get("id") if isinstance(term, dict) else term
            if not isinstance(identifier, str) or not re.fullmatch(r"HP:\d{7}", identifier):
                raise ValueError(f"{disease}: malformed HPO identifier {identifier!r}")
            identifiers.add(identifier)
        for identifier in identifiers:
            frequency[identifier] = frequency.get(identifier, 0) + 1
    return {term: -math.log(count / len(records)) for term, count in sorted(frequency.items())}

def mean_information_content(terms, scores):
    terms = sorted(set(terms))
    if not terms:
        return 0.0
    missing = set(terms) - scores.keys()
    if missing:
        raise ValueError(f"Graph phenotypes missing from IC source: {sorted(missing)}")
    values = [scores[term] for term in terms]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("IC scores must be finite and nonnegative")
    # A valid IC of zero MUST contribute to the denominator.
    return sum(values) / len(values)
