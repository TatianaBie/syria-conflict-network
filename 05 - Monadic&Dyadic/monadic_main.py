"""Build the monadic conflict dataset from the cleaned source files.

This script mirrors the logic in the Monadic - for Rmeltt_output notebook,
while replacing the hard-coded Dropbox path with the repo-wide project path
configuration and adding explicit CLI parameters: --dataset and --threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import CLEANED_DIR, DATA_DIR, MERGING_DIR, PROJECT_ROOT as REPO_ROOT

PROJECT_ROOT = REPO_ROOT

VALID_DATASETS = {"meltt", "acled", "ucdp"}
DEFAULT_DATASET = "meltt"
DEFAULT_THRESHOLD = 0.5
EXPANDED_ROW_ID_COLUMN = "_expanded_row_id"


def resolve_input_path(dataset: str, threshold: float) -> Path:
    """Resolve the canonical thresholded MELTT input for the monadic pipeline.

    The monadic workflow is anchored to a single shared file,
    dataset_threshold_{threshold}_meltt.csv. For the `acled` and `ucdp` cases,
    we keep the same source file and filter rows by the `dataset` column; for
    the `meltt` case we do not alter the rows.
    """
    dataset = dataset.lower()
    canonical_path = CLEANED_DIR / f"dataset_threshold_{threshold}_meltt.csv"
    if not canonical_path.exists():
        raise FileNotFoundError(
            f"Could not find the canonical monadic source file for dataset={dataset} and threshold={threshold}: {canonical_path}"
        )
    return canonical_path


def reshape_actor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape side_a/side_b columns into a name-by-year long format."""
    side_columns = [col for col in df.columns if col.startswith("Side_a_") or col.startswith("Side_b_")]
    if not side_columns:
        raise ValueError("The source dataset does not contain Side_a_ or Side_b_ actor columns.")

    reshaped = df.melt(
        id_vars=[EXPANDED_ROW_ID_COLUMN, "id", "year", "dataset"],
        value_vars=side_columns,
        var_name="column_indicator",
        value_name="name",
    )
    return reshaped


def build_monadic_dataset(dataset: str, threshold: float) -> pd.DataFrame:
    """Reproduce the monadic notebook logic for the selected source dataset.

    This project’s cleaned MELTT-expanded file contains the source rows under the
    actual 'dataset' column, with values such as acled and ucdp. The dataset
    argument therefore filters on that column for non-MELTT sources, while the
    MELTT case is treated as the baseline/no-op because the source data itself
    does not contain a meltt value in the dataset field.
    """
    input_path = resolve_input_path(dataset=dataset, threshold=threshold)
    df = pd.read_csv(input_path, low_memory=False)

    if "dataset" in df.columns and dataset.lower() != "meltt":
        df = df[df["dataset"].astype(str).str.lower() == dataset.lower()].copy()

    if df.empty:
        print(f"Warning: no rows found for dataset={dataset} in {input_path}; writing an empty output file.")
        return pd.DataFrame(columns=["name", "year", "Tot_Fights", "Ethnicity"])

    if "id" not in df.columns:
        df["id"] = df.index + 1

    # RF multilabel predictions are represented as separate physical rows. Keep
    # those alternatives independent even when they share the same source event
    # ID, while allowing duplicate actor cells within one expanded row to be
    # counted only once.
    df = df.reset_index(drop=True).copy()
    df[EXPANDED_ROW_ID_COLUMN] = df.index

    df_matching = pd.read_excel(MERGING_DIR / "Merging_names.xlsx", sheet_name="Renaming")
    df_ethnicity = pd.read_excel(MERGING_DIR / "Merging_ethnicity.xlsx", sheet_name="MAIN")

    df_reshaped = reshape_actor_data(df)
    df_reshaped = df_reshaped[df_reshaped["name"].isin(df_matching["Umbrella"])].copy()
    df_reshaped = df_reshaped[df_reshaped["year"] >= 2011].copy()

    valid_ids = (
        df_reshaped.groupby(EXPANDED_ROW_ID_COLUMN)["column_indicator"]
        .apply(lambda x: any(x.str.startswith("Side_a")) and any(x.str.startswith("Side_b")))
        .loc[lambda s: s]
        .index
    )
    df_filtered = df_reshaped[
        df_reshaped[EXPANDED_ROW_ID_COLUMN].isin(valid_ids)
    ].copy()
    df_filtered = df_filtered.drop_duplicates(
        subset=[EXPANDED_ROW_ID_COLUMN, "name"]
    )

    df_final = (
        df_filtered.groupby(["name", "year"]).size().reset_index(name="Tot_Fights").sort_values(["name", "year"])
    )
    monadic = df_final.merge(df_ethnicity[["name", "Ethnicity"]], on="name", how="left")
    monadic = monadic.sort_values(["name", "year"]).reset_index(drop=True)
    return monadic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce the monadic dataset from the cleaned MELTT/ACLED/UCDP inputs using the notebook logic."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(VALID_DATASETS),
        default=DEFAULT_DATASET,
        help="Source dataset to use: meltt, acled, or ucdp.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Probability threshold for the MELTT RF-expanded input. Default: 0.5.",
    )
    args = parser.parse_args()

    output_dir = DATA_DIR / "04 - Monadic&Dyadic_structure"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_output = build_monadic_dataset(dataset=args.dataset, threshold=args.threshold)
    output_path = output_dir / f"monadic_threshold_{args.threshold}_{args.dataset}.csv"
    df_output.to_csv(output_path, index=False)

    print(f"Saved monadic output to: {output_path}")
    print(f"Rows: {len(df_output)} | Columns: {list(df_output.columns)}")


if __name__ == "__main__":
    main()
