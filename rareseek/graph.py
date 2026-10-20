"""Validation of disease records used for local identifier mapping."""

def validate_graph(graph):
    if not isinstance(graph, dict) or not isinstance(graph.get('diseases'), list):
        raise ValueError('Graph must contain a diseases list')
    diseases = graph.get("diseases", [])
    if not diseases:
        raise ValueError("Graph has no diseases")
    seen = set()
    for disease in diseases:
        if not isinstance(disease, dict) or not isinstance(disease.get('id'), str) or not isinstance(disease.get('name'), str):
            raise ValueError('Disease IDs and names must be strings')
        if not disease.get("id") or disease["id"] in seen or not disease.get("name"):
            raise ValueError("Graph disease IDs must be unique; names are required")
        seen.add(disease["id"])
        for field in ("hpo_list", "genes", "variants", "aliases"):
            if not isinstance(disease.get(field, []), list) or any(not isinstance(v, str) for v in disease.get(field, [])):
                raise ValueError(f"Invalid graph field: {field}")
    return graph
