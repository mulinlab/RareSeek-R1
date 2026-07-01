from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence


BASE_DIR = Path("..data/raw")
INPUT_DIR = BASE_DIR / "Orphapacket"
OUTPUT_DIR = Path("../processed/Orphapacket")

RARE_DISEASE_SUMMARY_PATH = OUTPUT_DIR / "rare_disease_data_combined.tsv"
GENES_PATH = OUTPUT_DIR / "genes_extracted.tsv"
PARENTS_PATH = OUTPUT_DIR / "parents_extracted.tsv"
PHENOTYPES_PATH = OUTPUT_DIR / "phenotypes_extracted.tsv"
AVERAGE_AGE_PATH = OUTPUT_DIR / "AverageAgeOfOnsets.csv"
INHERITANCE_PATH = OUTPUT_DIR / "TypeOfInheritances.csv"


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def iter_json_files(directory: Path) -> Iterable[Path]:
    return sorted(directory.glob("*.json"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_get(data: Any, *keys: str) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def normalise_orpha_code(value: Any) -> str:
    """Return canonical ORPHA identifiers such as ORPHA:2584."""
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    upper = text.upper()
    if upper.startswith("ORPHA:"):
        suffix = text.split(":", 1)[1].strip()
        return f"ORPHA:{suffix}" if suffix else "ORPHA:"
    if upper.startswith("ORPHA"):
        suffix = text[5:].lstrip(":").strip()
        return f"ORPHA:{suffix}" if suffix else "ORPHA:"
    return f"ORPHA:{text}"


def join_non_empty(values: Iterable[Any], separator: str = "|") -> str:
    return separator.join(str(value) for value in values if value not in (None, ""))


def write_delimited(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str], delimiter: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def extract_orphapacket(data: dict[str, Any]) -> dict[str, Any] | None:
    orphapacket = data.get("Orphapacket")
    return orphapacket if isinstance(orphapacket, dict) else None


def collect_external_reference(ref: dict[str, Any]) -> str:
    source = str(ref.get("Source", "")).strip()
    reference = str(ref.get("Reference", "")).strip()
    if source and reference:
        return f"{source}:{reference}"
    return source or reference


def collect_text_section(orphapacket: dict[str, Any]) -> str:
    text_section = orphapacket.get("TextSection")
    if not isinstance(text_section, dict):
        return ""

    text_type = str(text_section.get("TextSectionType", "")).strip()
    contents = str(text_section.get("Contents", "")).strip()
    if text_type and contents:
        return f"{text_type}:{contents}"
    return text_type or contents


def process_rare_disease_summary() -> None:
    rows: list[dict[str, Any]] = []

    for json_path in iter_json_files(INPUT_DIR):
        try:
            data = load_json(json_path)
            orphapacket = extract_orphapacket(data)
            if not orphapacket:
                LOGGER.warning("Skipping %s: missing Orphapacket block", json_path.name)
                continue

            synonyms = [
                item.get("Synonym", "")
                for item in orphapacket.get("Synonyms", [])
                if isinstance(item, dict)
            ]
            ext_refs = [
                collect_external_reference(item)
                for item in orphapacket.get("ExternalReferences", [])
                if isinstance(item, dict)
            ]

            rows.append(
                {
                    "Label": orphapacket.get("Label", ""),
                    "ORPHAcode": normalise_orpha_code(orphapacket.get("ORPHAcode", "")),
                    "Synonyms": join_non_empty(synonyms),
                    "ExternalReferences": join_non_empty(ext_refs),
                    "TextSection": collect_text_section(orphapacket),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Error processing %s: %s", json_path.name, exc)

    write_delimited(
        RARE_DISEASE_SUMMARY_PATH,
        rows,
        ["Label", "ORPHAcode", "Synonyms", "ExternalReferences", "TextSection"],
        "\t",
    )
    LOGGER.info("Saved %d rows to %s", len(rows), RARE_DISEASE_SUMMARY_PATH)


def process_genes() -> None:
    rows: list[dict[str, Any]] = []
    max_gene_count = 0

    for json_path in iter_json_files(INPUT_DIR):
        try:
            data = load_json(json_path)
            orphapacket = extract_orphapacket(data)
            if not orphapacket:
                LOGGER.warning("Skipping %s: missing Orphapacket block", json_path.name)
                continue

            genes = orphapacket.get("Genes", [])
            if not isinstance(genes, list):
                genes = []

            row: dict[str, Any] = {
                "ORPHAcode": normalise_orpha_code(orphapacket.get("ORPHAcode", "")),
                "Disease": orphapacket.get("Label", ""),
            }

            gene_count = 0
            for index, gene_info in enumerate(genes, start=1):
                if not isinstance(gene_info, dict):
                    continue

                gene = gene_info.get("Gene", {})
                if not isinstance(gene, dict):
                    continue

                gene_count += 1
                row[f"gene_Symbol{index}"] = gene.get("Symbol", "")
                row[f"gene_Name{index}"] = gene.get("Name", "")
                row[f"gene_ExternalReferences{index}"] = join_non_empty(
                    collect_external_reference(ref)
                    for ref in gene.get("ExternalReferences", [])
                    if isinstance(ref, dict)
                )
                row[f"gene_DisorderGeneAssociationType{index}"] = gene.get(
                    "DisorderGeneAssociationType", ""
                )

            max_gene_count = max(max_gene_count, gene_count)
            rows.append(row)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Error processing %s: %s", json_path.name, exc)

    fieldnames = ["ORPHAcode", "Disease"]
    for index in range(1, max_gene_count + 1):
        fieldnames.extend(
            [
                f"gene_Symbol{index}",
                f"gene_Name{index}",
                f"gene_ExternalReferences{index}",
                f"gene_DisorderGeneAssociationType{index}",
            ]
        )

    write_delimited(GENES_PATH, rows, fieldnames, "\t")
    LOGGER.info("Saved %d rows to %s", len(rows), GENES_PATH)


def process_parents() -> None:
    rows: list[dict[str, Any]] = []
    max_parent_count = 0

    for json_path in iter_json_files(INPUT_DIR):
        try:
            data = load_json(json_path)
            orphapacket = extract_orphapacket(data)
            if not orphapacket:
                LOGGER.warning("Skipping %s: missing Orphapacket block", json_path.name)
                continue

            parents = orphapacket.get("Parents", [])
            if not isinstance(parents, list):
                parents = []

            row: dict[str, Any] = {
                "ORPHAcode": normalise_orpha_code(orphapacket.get("ORPHAcode", "")),
                "Disease": orphapacket.get("Label", ""),
            }

            parent_count = 0
            for parent_info in parents:
                if not isinstance(parent_info, dict):
                    continue

                parent_list = parent_info.get("Parent", [])
                if not isinstance(parent_list, list) or not parent_list:
                    continue

                parent = parent_list[0] if isinstance(parent_list[0], dict) else {}
                parent_count += 1
                row[f"parent_ORPHAcode{parent_count}"] = normalise_orpha_code(parent.get("ORPHAcode", ""))
                row[f"parent_Label{parent_count}"] = parent.get("Label", "")

            max_parent_count = max(max_parent_count, parent_count)
            rows.append(row)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Error processing %s: %s", json_path.name, exc)

    fieldnames = ["ORPHAcode", "Disease"]
    for index in range(1, max_parent_count + 1):
        fieldnames.extend([f"parent_ORPHAcode{index}", f"parent_Label{index}"])

    write_delimited(PARENTS_PATH, rows, fieldnames, "\t")
    LOGGER.info("Saved %d rows to %s", len(rows), PARENTS_PATH)


def process_phenotypes() -> None:
    rows: list[dict[str, Any]] = []
    max_phenotype_count = 0

    for json_path in iter_json_files(INPUT_DIR):
        try:
            data = load_json(json_path)
            orphapacket = extract_orphapacket(data)
            if not orphapacket:
                LOGGER.warning("Skipping %s: missing Orphapacket block", json_path.name)
                continue

            phenotypes = orphapacket.get("Phenotypes", [])
            if not isinstance(phenotypes, list):
                phenotypes = []

            row: dict[str, Any] = {
                "ORPHAcode": normalise_orpha_code(orphapacket.get("ORPHAcode", "")),
                "Disease": orphapacket.get("Label", ""),
            }

            phenotype_count = 0
            for phenotype_info in phenotypes:
                if not isinstance(phenotype_info, dict):
                    continue

                phenotype = phenotype_info.get("Phenotype", {})
                if not isinstance(phenotype, dict):
                    continue

                phenotype_count += 1
                row[f"phenotype_HPOId{phenotype_count}"] = phenotype.get("HPOId", "")
                row[f"phenotype_HPOTerm{phenotype_count}"] = phenotype.get("HPOTerm", "")
                row[f"phenotype_HPOFrequency{phenotype_count}"] = phenotype.get("HPOFrequency", "")

            max_phenotype_count = max(max_phenotype_count, phenotype_count)
            rows.append(row)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Error processing %s: %s", json_path.name, exc)

    fieldnames = ["ORPHAcode", "Disease"]
    for index in range(1, max_phenotype_count + 1):
        fieldnames.extend(
            [
                f"phenotype_HPOId{index}",
                f"phenotype_HPOTerm{index}",
                f"phenotype_HPOFrequency{index}",
            ]
        )

    write_delimited(PHENOTYPES_PATH, rows, fieldnames, "\t")
    LOGGER.info("Saved %d rows to %s", len(rows), PHENOTYPES_PATH)


def extract_sequence_values(sequence: Any, nested_key: str | None = None) -> list[str]:
    values: list[str] = []

    if not isinstance(sequence, list):
        return values

    for item in sequence:
        if isinstance(item, dict):
            if nested_key and isinstance(item.get(nested_key), dict):
                nested = item[nested_key]
                value = nested.get("value")
                if value not in (None, ""):
                    values.append(str(value))
            else:
                value = item.get("value")
                if value not in (None, ""):
                    values.append(str(value))
        elif isinstance(item, str) and item.strip():
            values.append(item)

    return values


def process_average_age_of_onsets() -> None:
    rows: list[dict[str, Any]] = []

    for json_path in iter_json_files(INPUT_DIR):
        try:
            data = load_json(json_path)
            orphapacket = extract_orphapacket(data)
            if not orphapacket:
                LOGGER.warning("Skipping %s: missing Orphapacket block", json_path.name)
                continue

            age_values = extract_sequence_values(orphapacket.get("AverageAgeOfOnsets"), "AverageAgeOfOnset")
            row: dict[str, Any] = {
                "ORPHAcode": normalise_orpha_code(orphapacket.get("ORPHAcode", "")),
                "Disease": orphapacket.get("Label", ""),
            }
            for index, value in enumerate(age_values, start=1):
                row[f"AverageAgeOfOnset_value{index}"] = value

            rows.append(row)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Error processing %s: %s", json_path.name, exc)

    max_age_count = max(
        (len([key for key in row if key.startswith("AverageAgeOfOnset_value")]) for row in rows),
        default=0,
    )
    fieldnames = ["ORPHAcode", "Disease"] + [f"AverageAgeOfOnset_value{i}" for i in range(1, max_age_count + 1)]
    write_delimited(AVERAGE_AGE_PATH, rows, fieldnames, ",")
    LOGGER.info("Saved %d rows to %s", len(rows), AVERAGE_AGE_PATH)


def process_inheritances() -> None:
    rows: list[dict[str, Any]] = []

    for json_path in iter_json_files(INPUT_DIR):
        try:
            data = load_json(json_path)
            orphapacket = extract_orphapacket(data)
            if not orphapacket:
                LOGGER.warning("Skipping %s: missing Orphapacket block", json_path.name)
                continue

            type_of_inheritances = orphapacket.get("TypeOfInheritances", {})
            if isinstance(type_of_inheritances, dict):
                inheritance_values = extract_sequence_values(type_of_inheritances.get("TypeOfInheritance"))
            elif isinstance(type_of_inheritances, list):
                inheritance_values = extract_sequence_values(type_of_inheritances)
            else:
                inheritance_values = []

            row: dict[str, Any] = {
                "ORPHAcode": normalise_orpha_code(orphapacket.get("ORPHAcode", "")),
                "Label": orphapacket.get("Label", ""),
            }
            for index, value in enumerate(inheritance_values, start=1):
                row[f"TypeOfInheritance_value{index}"] = value

            rows.append(row)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Error processing %s: %s", json_path.name, exc)

    max_inheritance_count = max(
        (len([key for key in row if key.startswith("TypeOfInheritance_value")]) for row in rows),
        default=0,
    )
    fieldnames = ["ORPHAcode", "Label"] + [f"TypeOfInheritance_value{i}" for i in range(1, max_inheritance_count + 1)]
    write_delimited(INHERITANCE_PATH, rows, fieldnames, ",")
    LOGGER.info("Saved %d rows to %s", len(rows), INHERITANCE_PATH)


def main() -> None:
    ensure_output_dir()
    process_rare_disease_summary()
    process_genes()
    process_parents()
    process_phenotypes()
    process_average_age_of_onsets()
    process_inheritances()


if __name__ == "__main__":
    main()

