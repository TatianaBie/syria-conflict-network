"""Prepare the manual pairwise relationship dataset.

This script loads the historical manual relationship table, expands each
relationship through the selected end year, and writes a standardized yearly
pairwise output table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import PROJECT_ROOT as REPO_ROOT

PROJECT_ROOT = REPO_ROOT

DEFAULT_END_YEAR = 2024
DEFAULT_SOURCE_NAME = "Manual"


def load_manual_relationships(input_path: Path) -> pd.DataFrame:
    """Load the legacy manual relationship file and drop incomplete rows."""
    df = pd.read_excel(input_path)

    if "start_year" not in df.columns:
        raise ValueError("Expected 'start_year' column in the manual relationship file.")

    df = df[df["start_year"] != "x"].copy()
    df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")
    df = df.dropna(subset=["start_year"]).copy()
    return df


def expand_relationship_timeline(df: pd.DataFrame, end_year: int) -> list[dict]:
    """Repeat each relationship for each year from its start year through the end year."""
    expanded_rows: list[dict] = []
    for _, row in df.iterrows():
        start_year = int(row["start_year"])
        for year in range(start_year, end_year + 1):
            new_row = row.copy()
            new_row["year"] = year
            expanded_rows.append(new_row.to_dict())
    return expanded_rows


def normalize_relationships(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Rename fields and map the multi-coded relationship flags to standard names."""
    df = df.rename(
        columns={
            "Primary Group": "group1_name",
            "Other Group": "group2_name",
            "Relationship": "status",
        }
    )

    df["status"] = "Unknown"
    df.loc[df["allied_multi"] == 1, "status"] = "Allies"
    df.loc[df["enemy_multi"] == 1, "status"] = "Enemies"

    df["source"] = source_name
    output_columns = ["group1_name", "group2_name", "year", "status", "source"]
    return df[output_columns].copy()


def run_pipeline(
    input_path: Path,
    output_path: Path,
    end_year: int,
    source_name: str = DEFAULT_SOURCE_NAME,
) -> pd.DataFrame:
    """Run the full manual pairwise preparation pipeline."""
    manual_df = load_manual_relationships(input_path)
    expanded_rows = expand_relationship_timeline(manual_df, end_year=end_year)
    expanded_df = pd.DataFrame(expanded_rows)
    cleaned_df = normalize_relationships(expanded_df, source_name=source_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Saved manual pairwise output to: {output_path}")
    print(cleaned_df.head(10).to_string(index=False))
    return cleaned_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the legacy manual relationship file into a standard yearly pairwise dataset."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=PROJECT_ROOT / "01 - Data" / "01 - Source_Data" / "Manual" / "Manual.xlsx",
        help="Path to the manual relationship Excel file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "01 - Data" / "03 - Pairwise_connections" / "Manual-connections.csv",
        help="Output CSV path for the cleaned yearly pairwise table.",
    )
    parser.add_argument(
        "--source-name",
        default=DEFAULT_SOURCE_NAME,
        help="Source label written to the output. Default: Manual.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help="Final year to which each relationship should be expanded. Default: 2024.",
    )
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        end_year=args.end_year,
        source_name=args.source_name,
    )


if __name__ == "__main__":
    main()
