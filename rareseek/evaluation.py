"""Exact and hierarchical accuracy on the same, fixed gold cohort."""
from .io import by_id

def validate_predictions(groups):
    if not isinstance(groups, list) or any(
        not isinstance(group, list) or any(not isinstance(code, str) or not code for code in group)
        for group in groups
    ):
        raise ValueError("predict_code must be a ranked list of lists of nonempty codes")
    return groups

def expanded_targets(labels, hierarchy):
    # Hierarchy is child code -> direct parents. Exact codes are always retained.
    children = {}
    for child, parents in hierarchy.items():
        if not isinstance(parents, list) or any(not isinstance(p, str) for p in parents):
            raise ValueError("Hierarchy values must be lists of parent codes")
        for parent in parents:
            children.setdefault(parent, set()).add(child)
    targets = set(labels)
    for label in labels:
        if not label.startswith("ORPHA:"):
            continue
        targets.update(hierarchy.get(label, []))
        direct = children.get(label, set())
        targets.update(direct)
        for child in direct:
            targets.update(children.get(child, set()))
    return targets

def compile_hierarchy(hierarchy):
    children = {}
    for child, parents in hierarchy.items():
        if not isinstance(parents, list) or any(not isinstance(p, str) for p in parents):
            raise ValueError("Hierarchy values must be lists of parent codes")
        for parent in parents:
            children.setdefault(parent, set()).add(child)
    expanded = {}
    for code in set(hierarchy) | set(children):
        direct = children.get(code, set())
        expanded[code] = {code, *hierarchy.get(code, []), *direct}
        for child in direct:
            expanded[code].update(children.get(child, set()))
    return expanded

def evaluate(gold_rows, predictions, hierarchy=None, ks=(1, 3, 5, 10, 20)):
    gold, predicted = by_id(gold_rows), by_id(predictions)
    if not gold:
        raise ValueError("Gold cohort is empty")
    if set(predicted) - set(gold):
        raise ValueError("Predictions contain sample IDs outside the gold cohort")
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("k values must be positive integers")
    exact = {str(k): 0 for k in ks}
    hierarchical = exact.copy()
    expansion = compile_hierarchy(hierarchy or {})
    details = []
    for sid, row in gold.items():
        labels = row.get("true_label")
        if labels is None and row.get("orphacode"):
            labels = row["orphacode"]
            if isinstance(labels, str):
                labels = [labels]
        if not isinstance(labels, list) or not labels or any(not isinstance(x, str) or not x for x in labels):
            raise ValueError(f"{sid}: supply orphacode or a nonempty true_label list")
        pred = predicted.get(sid, {})
        groups = validate_predictions(pred.get("predict_code", []))
        status = "missing" if sid not in predicted else ("empty" if not any(groups) else "predicted")
        targets = set(labels)
        for label in labels:
            if label.startswith("ORPHA:"):
                targets.update(expansion.get(label, set()))
        detail = {"sample_id": sid, "status": status, "exact": {}, "hierarchical": {}}
        for k in ks:
            codes = {code for group in groups[:k] for code in group}
            e, h = bool(codes & set(labels)), bool(codes & targets)
            exact[str(k)] += int(e)
            hierarchical[str(k)] += int(h)
            detail["exact"][str(k)] = e
            detail["hierarchical"][str(k)] = h
        details.append(detail)
    return {
        "total": len(gold), "missing": sum(d["status"] == "missing" for d in details),
        "empty": sum(d["status"] == "empty" for d in details),
        "exact": {k: {"hits": n, "accuracy": n / len(gold)} for k, n in exact.items()},
        "hierarchical": {k: {"hits": n, "accuracy": n / len(gold)} for k, n in hierarchical.items()},
        "hierarchy_rule": "exact + direct parents + children + grandchildren; ORPHA only",
        "per_sample": details,
    }
