"""Local parser for numbered final diagnoses; no clinical text leaves the machine."""
import argparse
from rareseek.io import path, read_jsonl, write_jsonl
from rareseek.pipeline import parse_diagnoses
class RareDiseaseEntityExtractor:
    def extract_disease_entities(self, diagnostic_text):
        return parse_diagnoses(diagnostic_text)
def process_diagnostic_dataset(input_filepath, output_filepath):
    rows = read_jsonl(path(input_filepath))
    for row in rows:
        if "predict_result" not in row and "diagnosis_results" not in row:
            raise ValueError("Missing predict_result (legacy diagnosis_results accepted)")
        row["extracted_diseases"] = parse_diagnoses(row.get("predict_result", row.get("diagnosis_results", "")))
    write_jsonl(path(output_filepath), rows)
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    process_diagnostic_dataset(a.input, a.output)
