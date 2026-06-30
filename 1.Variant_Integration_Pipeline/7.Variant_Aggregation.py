"""
Variant Aggregation and Phenotype Annotation Pipeline

This script aggregates chromosome-specific variant files into a unified dataset 
and annotates ClinVar-derived variants with their corresponding standard 
Phenotype Identifiers (PhenotypeIDS) by mapping genomic coordinates.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict

# Configure pipeline logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def aggregate_variant_chunks(directory_path: Path, file_pattern: str) -> pd.DataFrame:
    """
    Scans the specified directory for chunked variant files and concatenates 
    them into a single comprehensive DataFrame.

    Args:
        directory_path (Path): Path to the directory containing variant chunks.
        file_pattern (str): Glob pattern to identify the target files.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing all variants.
        
    Raises:
        FileNotFoundError: If no files matching the pattern are found.
    """
    file_paths = list(directory_path.glob(file_pattern))
    
    if not file_paths:
        logger.error(f"No files matching '{file_pattern}' found in {directory_path}.")
        raise FileNotFoundError(f"Missing variant chunk files in {directory_path}")

    logger.info(f"Identified {len(file_paths)} variant files. Initiating aggregation...")
    
    # Read and concatenate all files, coercing everything to string to prevent dtype mismatches
    dataframes = [pd.read_csv(file, sep='\t', dtype=str) for file in file_paths]
    aggregated_df = pd.concat(dataframes, ignore_index=True)
    
    logger.info(f"Aggregation complete. Total variants loaded: {len(aggregated_df)}.")
    return aggregated_df


def annotate_phenotype_ids(
    aggregated_variants: pd.DataFrame, 
    clinvar_summary_path: Path
) -> pd.DataFrame:
    """
    Maps PhenotypeIDS from the official ClinVar summary to the aggregated dataset
    using a composite genomic coordinate key (Chr|Pos|Ref|Alt).

    Args:
        aggregated_variants (pd.DataFrame): The compiled variant dataset.
        clinvar_summary_path (Path): Path to the ClinVar variant_summary.txt file.

    Returns:
        pd.DataFrame: The annotated DataFrame with an appended 'PhenotypeIDS' column.
    """
    logger.info(f"Loading ClinVar summary reference from {clinvar_summary_path}...")
    clinvar_summary_df = pd.read_csv(clinvar_summary_path, sep="\t", dtype=str)

    # Construct the composite genomic key for the ClinVar reference dataset
    clinvar_summary_df["genomic_key"] = (
        clinvar_summary_df["Chromosome"] + "|" + 
        clinvar_summary_df["PositionVCF"] + "|" + 
        clinvar_summary_df["ReferenceAlleleVCF"] + "|" + 
        clinvar_summary_df["AlternateAlleleVCF"]
    )
    
    # Generate a mapping dictionary: {genomic_key: PhenotypeIDS}
    genomic_key_to_phenotype_map: Dict[str, str] = clinvar_summary_df.set_index("genomic_key")["PhenotypeIDS"].to_dict()

    # Initialize the target column with empty strings
    aggregated_variants["PhenotypeIDS"] = ""
    
    # Isolate rows originating from ClinVar for targeted annotation
    clinvar_source_mask = aggregated_variants["source"] == "clinvar"
    
    # Construct the composite genomic key for the target dataset
    target_genomic_keys = (
        aggregated_variants.loc[clinvar_source_mask, "chr"] + "|" +
        aggregated_variants.loc[clinvar_source_mask, "pos"] + "|" +
        aggregated_variants.loc[clinvar_source_mask, "ref"] + "|" +
        aggregated_variants.loc[clinvar_source_mask, "alt"]
    )

    logger.info("Mapping PhenotypeIDS to the aggregated dataset...")
    # Map the identifiers and handle any unmapped coordinates
    mapped_phenotypes = target_genomic_keys.map(genomic_key_to_phenotype_map)
    aggregated_variants.loc[clinvar_source_mask, "PhenotypeIDS"] = mapped_phenotypes.fillna("")
    
    return aggregated_variants


def main() -> None:
    """Main execution block for the variant annotation pipeline."""
    # Define file paths
    variant_directory = Path("variant")
    clinvar_summary_file = Path("../clinvar/variant_summary.txt")
    output_file = Path("merged_variants_with_phenotype.csv")

    try:
        # Step 1: Aggregate data chunks
        aggregated_df = aggregate_variant_chunks(
            directory_path=variant_directory, 
            file_pattern="*.HGMD_and_clinvar.csv"
        )

        # Step 2: Annotate with ClinVar PhenotypeIDS
        annotated_df = annotate_phenotype_ids(
            aggregated_variants=aggregated_df, 
            clinvar_summary_path=clinvar_summary_file
        )

        # Step 3: Export the finalized dataset
        logger.info(f"Exporting finalized dataset to {output_file}...")
        annotated_df.to_csv(output_file, sep="\t", index=False)
        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline terminated unexpectedly due to an error: {e}")


if __name__ == "__main__":
    main()