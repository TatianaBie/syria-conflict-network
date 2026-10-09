"""Prepare Stanford pairwise group relationships for downstream analysis.

This script standardizes the Stanford relationship data into a reproducible,
parameterized pipeline. It supports both the Syria-wide network and the
Aleppo-specific network and writes a cleaned pairwise-output table into the
project data folder.
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

VALID_MAPS = {"syria", "aleppo"}
DEFAULT_MAP = "aleppo"
DEFAULT_END_YEAR = 2024


def get_input_path(map_name: str) -> Path:
    """Return the Stanford source CSV matching the requested map."""
    source_dir = PROJECT_ROOT / "01 - Data" / "01 - Source_Data" / "Stanford"
    if map_name == "syria":
        return source_dir / "stanford_network_240707.csv"
    if map_name == "aleppo":
        return source_dir / "stanford_aleppo_from2011_20241110.csv"
    raise ValueError(f"Unsupported Stanford map: {map_name}")


def find_duplicate_groups(df: pd.DataFrame) -> pd.Series:
    """Return the pair counts for repeated Stanford group relationships."""
    pair_key = df.apply(lambda row: tuple(sorted([row["group1_name"], row["group2_name"]])), axis=1)
    return pair_key.value_counts()


def drop_duplicate_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Remove true duplicate ties while preserving distinct dyad-year-status entries."""
    df = df.copy()
    df["group1_name"] = df["group1_name"].astype(str).str.strip()
    df["group2_name"] = df["group2_name"].astype(str).str.strip()

    pair_key = df.apply(
        lambda row: tuple(sorted([row["group1_name"], row["group2_name"]])),
        axis=1,
    )
    df["pair_key"] = pair_key

    n_before = len(df)
    df = df.drop_duplicates(subset=["pair_key", "year", "status"], keep="first").copy()
    n_after = len(df)
    print(f"Removed {n_before - n_after} duplicate Stanford rows via drop_duplicates().")
    return df.drop(columns=["pair_key"])


def expand_years(df: pd.DataFrame, end_year: int = DEFAULT_END_YEAR) -> pd.DataFrame:
    """Repeat each row for each year in the observed range up to the target year."""
    expanded_rows: list[dict] = []
    for _, row in df.iterrows():
        start_year = int(row["year"])
        for year in range(start_year, end_year + 1):
            new_row = row.copy()
            new_row["year"] = year
            expanded_rows.append(new_row.to_dict())
    return pd.DataFrame(expanded_rows)


def clean_pairwise_data(df: pd.DataFrame, map_name: str) -> pd.DataFrame:
    """Normalize statuses and keep the cleaned pairwise relationship set."""
    df = df.copy()
    df["source"] = f"Stanford-{map_name}"
    result = df[["group1_name", "group2_name", "year", "status", "source"]].copy()
    result = result[result["status"].isin(["Affiliates", "Allies", "Rivals"])].copy()
    result.loc[result["status"] == "Rivals", "status"] = "Enemies"
    result.loc[result["status"] == "Affiliates", "status"] = "Allies"
    return result


def main(map_name: str, end_year: int) -> None:
    """Load, clean, and export the Stanford network for the selected region."""
    input_path = get_input_path(map_name)
    output_dir = PROJECT_ROOT / "01 - Data" / "03 - Pairwise_connections"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, index_col=0)
    duplicate_counts = find_duplicate_groups(df)
    print(f"Duplicate pair count: {(duplicate_counts > 1).sum()}")
    if not duplicate_counts.empty:
        print(duplicate_counts[duplicate_counts > 1].head(10).to_string())

    df = drop_duplicate_relationships(df)
    df_expanded = expand_years(df, end_year=end_year)
    cleaned = clean_pairwise_data(df_expanded, map_name)

    output_path = output_dir / f"Stanford_{map_name.title()}-connections.csv"
    cleaned.to_csv(output_path, index=False)

    print(f"Saved cleaned Stanford output to: {output_path}")
    print(cleaned.head().to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a cleaned Stanford pairwise relationship table for downstream analysis."
    )
    parser.add_argument(
        "--map",
        choices=sorted(VALID_MAPS),
        default=DEFAULT_MAP,
        help="Choose 'aleppo' or 'syria'. Default: aleppo.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help="Last year used when expanding the relationship timeline. Default: 2024.",
    )
    args = parser.parse_args()

    main(args.map, args.end_year)
