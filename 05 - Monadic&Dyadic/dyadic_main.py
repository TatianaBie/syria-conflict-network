"""Build the combined dyadic dataset from the curated pairwise sources.

This script reproduces the logic in the Dyadic - MAIN notebook using the
updated pairwise outputs. It keeps the data pipeline explicit and exposes the
main runtime controls as CLI arguments: --dataset and --threshold.

The script reads the dataset-specific MELTT pairwise file for the selected
source (meltt, acled, or ucdp) and combines it with the non-MELTT source files
used in the notebook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DATA_DIR, MERGING_DIR, PROJECT_ROOT as REPO_ROOT

PROJECT_ROOT = REPO_ROOT

VALID_DATASETS = {"meltt", "acled", "ucdp"}
DEFAULT_DATASET = "meltt"
DEFAULT_THRESHOLD = 0.5


def normalize_source_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the source labels across pairwise outputs."""
    df = df.copy()
    if "source" not in df.columns:
        return df

    source_mapping = {
        "stanford-aleppo": "Stanford-Aleppo",
        "stanford-syria": "Stanford-Syria",
        "oldstanford": "OldStanford",
        "old-stanford": "OldStanford",
        "wiki": "Wiki",
        "manual": "Manual",
        "meltt": "meltt",
        "acled": "meltt",
        "ucdp": "meltt",
    }

    df["source"] = df["source"].astype(str).str.strip()
    df["source"] = df["source"].str.lower().map(source_mapping).fillna(df["source"])
    return df


def resolve_wiki_path() -> Path:
    """Resolve the single combined Wiki file used by the Dropbox workflow."""
    pairwise_dir = DATA_DIR / "03 - Pairwise_connections"
    wiki_path = pairwise_dir / "ConflictWiki-connections_withcounts_new.csv"
    if not wiki_path.exists():
        raise FileNotFoundError(
            f"Expected Dropbox-compatible Wiki pairwise file not found: {wiki_path}"
        )
    return wiki_path


def resolve_meltt_path(dataset: str, threshold: float) -> Path:
    """Resolve the dataset-specific MELTT pairwise input file."""
    pairwise_dir = DATA_DIR / "03 - Pairwise_connections"
    candidate = pairwise_dir / f"meltt_pairwise_{dataset}_{threshold}.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Expected MELTT pairwise file not found: {candidate}")
    return candidate


def load_pairwise_inputs(dataset: str, threshold: float) -> dict[str, pd.DataFrame]:
    """Load all non-MELTT pairwise files plus the selected MELTT dataset."""
    pairwise_dir = DATA_DIR / "03 - Pairwise_connections"
    frames = {
        "Manual": pd.read_csv(pairwise_dir / "Manual-connections.csv"),
        "OldStanford": pd.read_csv(pairwise_dir / "OldStanford-connections.csv"),
        "Stanford-Aleppo": pd.read_csv(pairwise_dir / "Stanford_Aleppo-connections.csv"),
        "Stanford-Syria": pd.read_csv(pairwise_dir / "Stanford_Syria-connections.csv"),
        "Wiki": pd.read_csv(resolve_wiki_path()),
        "meltt": pd.read_csv(resolve_meltt_path(dataset=dataset, threshold=threshold)),
    }

    for key, df in frames.items():
        if "source" not in df.columns:
            df["source"] = key
        elif key == "meltt":
            df["source"] = "meltt"
        frames[key] = normalize_source_names(df)

    return frames


def rename_by_umbrella(df: pd.DataFrame, mapping_file: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Map source-specific names to the umbrella name used throughout the dyadic dataset."""
    df = df.copy()

    source_map = {
        "Manual": "Manual_Name",
        "OldStanford": "Manual_Name",
        "Stanford-Aleppo": "Stanford_Name",
        "Stanford-Syria": "Stanford_Name",
        "Wiki": "Wiki_Name",
        "meltt": "ACLED_Name",
    }

    mapping_column = source_map.get(source_name)
    if mapping_column is None:
        return df

    mapping = mapping_file[[mapping_column, "Umbrella"]].dropna()
    mapping_dict = dict(zip(mapping[mapping_column].astype(str), mapping["Umbrella"].astype(str)))

    for col in ["group1_name", "group2_name"]:
        df[col] = df[col].astype(str).replace(mapping_dict).fillna(df[col])

    return df


def prepare_data(dataset: str, threshold: float) -> pd.DataFrame:
    """Load, normalize, and combine all relevant source tables."""
    pairwise_frames = load_pairwise_inputs(dataset=dataset, threshold=threshold)
    matching = pd.read_excel(MERGING_DIR / "Merging_names.xlsx", sheet_name="Renaming")

    combined = []
    for source_name, df in pairwise_frames.items():
        cleaned = rename_by_umbrella(df, matching, source_name)
        if not cleaned.empty:
            combined.append(cleaned)

    if not combined:
        return pd.DataFrame(columns=["year", "status", "group1_name", "group2_name", "source", "Count"])

    df = pd.concat(combined, ignore_index=True, sort=False)
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    return df


def filter_valid_names(df: pd.DataFrame, matching: pd.DataFrame) -> pd.DataFrame:
    """Keep only dyads whose names are represented in the umbrella scope."""
    valid_names = set(matching["Umbrella"].dropna().astype(str))
    filtered = df[df["group1_name"].astype(str).isin(valid_names) & df["group2_name"].astype(str).isin(valid_names)].copy()
    return filtered


def apply_manual_stanford_limits(df: pd.DataFrame, limits_path: Path) -> pd.DataFrame:
    """Apply the notebook's validity-window filters to manual and Stanford rows."""
    limits = pd.read_csv(limits_path)
    source_filter = ["Manual", "Stanford-Syria", "Stanford-Aleppo", "OldStanford"]

    filtered_df = df[df["source"].isin(source_filter)].copy()
    if filtered_df.empty:
        return df

    df_merged1 = filtered_df.merge(limits, left_on="group1_name", right_on="name", how="inner", suffixes=("", "_group1"))
    df_merged1 = df_merged1.rename(columns={"min": "min_group1", "max": "max_group1"}).drop(columns=["name"])

    df_merged2 = df_merged1.merge(limits, left_on="group2_name", right_on="name", how="inner", suffixes=("", "_group2"))
    df_merged2 = df_merged2.rename(columns={"min": "min_group2", "max": "max_group2"}).drop(columns=["name"])

    valid_rows = df_merged2[
        (df_merged2["year"] >= df_merged2["min_group1"]) & (df_merged2["year"] <= df_merged2["max_group1"])
        & (df_merged2["year"] >= df_merged2["min_group2"]) & (df_merged2["year"] <= df_merged2["max_group2"])
    ]

    result = valid_rows[["year", "status", "group1_name", "group2_name", "source", "Count"]].copy()
    rest_df = df[~df["source"].isin(source_filter)].copy()
    return pd.concat([result, rest_df], ignore_index=True, sort=False)


def run_pipeline(dataset: str, threshold: float) -> pd.DataFrame:
    """Run the Dyadic notebook pipeline for the selected MELTT subset."""
    if dataset not in VALID_DATASETS:
        raise ValueError(f"dataset must be one of {sorted(VALID_DATASETS)}")

    matching = pd.read_excel(MERGING_DIR / "Merging_names.xlsx", sheet_name="Renaming")
    df = prepare_data(dataset=dataset, threshold=threshold)
    df = filter_valid_names(df, matching)

    df[["group1_name", "group2_name"]] = df.apply(
        lambda row: pd.Series(sorted([row["group1_name"], row["group2_name"]])), axis=1
    )
    df = df[df["group1_name"] != df["group2_name"]].copy()

    df = apply_manual_stanford_limits(df, DATA_DIR / "02 - Source_Data-cleaned" / "MAIN_UCDP+Acled_Groups_min&max.csv")
    df = df.drop_duplicates().sort_values(["group1_name", "group2_name", "year", "status", "source"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce the dyadic dataset from the updated pairwise information using the Dyadic - MAIN logic."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(VALID_DATASETS),
        default=DEFAULT_DATASET,
        help="Pairwise MELTT dataset to use: meltt, acled, or ucdp.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="RF threshold for the selected MELTT pairwise file. Default: 0.5.",
    )
    args = parser.parse_args()

    output_dir = DATA_DIR / "04 - Monadic&Dyadic_structure"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_output = run_pipeline(dataset=args.dataset, threshold=args.threshold)
    output_path = output_dir / f"dyadic_threshold_{args.threshold}_{args.dataset}.csv"
    df_output.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
