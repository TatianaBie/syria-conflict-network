"""Build MELTT pairwise relationships from the RF-expanded source data.

This script implements the MELTT pairwise construction pipeline in a reusable,
production-ready format. It resolves project paths from a central root config,
constructs enemy and ally dyads by year, and exposes the main runtime controls
via CLI arguments: `--dataset` and `--threshold`.
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import CLEANED_DIR, MERGING_DIR, PROJECT_ROOT as REPO_ROOT

PROJECT_ROOT = REPO_ROOT

VALID_DATASETS = {"meltt", "ucdp", "acled"}
DEFAULT_DATASET = "meltt"
DEFAULT_THRESHOLD = 0.5
ROW_ID_COLUMN = "_expanded_row_id"


def find_numbered_side_columns(df: pd.DataFrame, side: str) -> list[str]:
    """Return every numbered actor column for one event side in numeric order."""
    pattern = re.compile(rf"^Side_{re.escape(side)}_(\d+)$")
    numbered_columns = []
    for column in df.columns:
        match = pattern.fullmatch(column)
        if match:
            numbered_columns.append((int(match.group(1)), column))

    if not numbered_columns:
        raise ValueError(f"No numbered Side_{side}_* columns found in MELTT input.")

    return [column for _, column in sorted(numbered_columns)]


def resolve_input_path(threshold: float) -> Path:
    """Resolve the thresholded MELTT source file for the selected threshold."""
    input_path = CLEANED_DIR / f"dataset_threshold_{threshold}_meltt.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Required thresholded MELTT source file not found: {input_path}"
        )
    return input_path


def build_fights(df: pd.DataFrame) -> pd.DataFrame:
    """Create Enemy relationships from all Side A / Side B dyads in a conflict event."""
    side_a_columns = find_numbered_side_columns(df, "a")
    side_b_columns = find_numbered_side_columns(df, "b")

    # RF expansion can produce several physical rows with the same original
    # event ID. Keep those alternatives independent during pair construction.
    df = df.reset_index(drop=True).copy()
    df[ROW_ID_COLUMN] = df.index

    df_side_a = df.melt(
        id_vars=[ROW_ID_COLUMN, "id", "year"],
        value_vars=side_a_columns,
        var_name="side_a_column",
        value_name="group1_name",
    )
    df_side_b = df.melt(
        id_vars=[ROW_ID_COLUMN, "id", "year"],
        value_vars=side_b_columns,
        var_name="side_b_column",
        value_name="group2_name",
    )

    df_side_a = (
        df_side_a.dropna(subset=["group1_name"])
        .drop_duplicates(subset=[ROW_ID_COLUMN, "group1_name"])
        .copy()
    )
    df_side_b = (
        df_side_b.dropna(subset=["group2_name"])
        .drop_duplicates(subset=[ROW_ID_COLUMN, "group2_name"])
        .copy()
    )

    df_merged = pd.merge(
        df_side_a,
        df_side_b,
        on=[ROW_ID_COLUMN, "id", "year"],
        how="inner",
    )
    df_fights = df_merged.copy()

    df_fights[["group1_name_sorted", "group2_name_sorted"]] = df_fights.apply(
        lambda row: pd.Series(sorted([row["group1_name"], row["group2_name"]])),
        axis=1,
    )

    df_fights = (
        df_fights.groupby(["group1_name_sorted", "group2_name_sorted", "year"])
        .size()
        .reset_index(name="Count")
    )
    df_fights["status"] = "Enemies"
    df_fights = df_fights.rename(
        columns={
            "group1_name_sorted": "group1_name",
            "group2_name_sorted": "group2_name",
        }
    )
    return df_fights[["group1_name", "group2_name", "year", "status", "Count"]]


def build_cooperations(df: pd.DataFrame) -> pd.DataFrame:
    """Create Ally relationships from pairwise combinations within each side."""
    empty_return = pd.DataFrame(columns=["group1_name", "group2_name", "year", "status", "Count"])

    side_a_columns = find_numbered_side_columns(df, "a")
    side_b_columns = find_numbered_side_columns(df, "b")

    # Preserve the physical RF-expanded rows as separate alternatives even
    # when they share the same original event ID.
    df = df.reset_index(drop=True).copy()
    df[ROW_ID_COLUMN] = df.index
    df_side_a = df[[ROW_ID_COLUMN, "id", "year", *side_a_columns]].copy()
    df_side_b = df[[ROW_ID_COLUMN, "id", "year", *side_b_columns]].copy()

    df_a = df_side_a.melt(
        id_vars=[ROW_ID_COLUMN, "id", "year"],
        value_vars=side_a_columns,
        var_name="side_a_column",
        value_name="Cooperation",
    ).dropna(subset=["Cooperation"])
    df_a = df_a.drop_duplicates(subset=[ROW_ID_COLUMN, "Cooperation"]).copy()

    df_b = df_side_b.melt(
        id_vars=[ROW_ID_COLUMN, "id", "year"],
        value_vars=side_b_columns,
        var_name="side_b_column",
        value_name="Cooperation",
    ).dropna(subset=["Cooperation"])
    df_b = df_b.drop_duplicates(subset=[ROW_ID_COLUMN, "Cooperation"]).copy()

    def extract_pairs(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["group1_name", "group2_name", "year"])

        pairs = (
            frame.groupby([ROW_ID_COLUMN, "id", "year"])["Cooperation"]
            .apply(lambda x: list(combinations(pd.unique(x), 2)))
            .explode()
            .dropna()
        )

        if pairs.empty:
            return pd.DataFrame(columns=["group1_name", "group2_name", "year"])

        tuples = list(pairs.tolist())
        pair_df = pd.DataFrame(tuples, columns=["group1_name", "group2_name"])
        pair_df["year"] = pairs.index.get_level_values("year").to_numpy()
        return pair_df[["group1_name", "group2_name", "year"]].reset_index(drop=True)

    df_pairs_a = extract_pairs(df_a)
    df_pairs_b = extract_pairs(df_b)

    if df_pairs_a.empty and df_pairs_b.empty:
        return empty_return

    df_cooperations = pd.concat([df_pairs_a, df_pairs_b], ignore_index=True)
    if df_cooperations.empty:
        return empty_return

    df_cooperations[["group1_name_sorted", "group2_name_sorted"]] = df_cooperations.apply(
        lambda row: pd.Series(sorted([row["group1_name"], row["group2_name"]])),
        axis=1,
    )

    df_cooperations = (
        df_cooperations.groupby(["group1_name_sorted", "group2_name_sorted", "year"])
        .size()
        .reset_index(name="Count")
    )
    df_cooperations["status"] = "Allies"
    df_cooperations = df_cooperations.rename(
        columns={
            "group1_name_sorted": "group1_name",
            "group2_name_sorted": "group2_name",
        }
    )
    return df_cooperations[["group1_name", "group2_name", "year", "status", "Count"]]


def filter_umbrella_matches(df: pd.DataFrame, umbrella_file: Path | None) -> pd.DataFrame:
    """Keep only dyads whose names are present in the umbrella renaming table.

    This follows the notebook logic but is kept explicit and optional so it does
    not silently act as a general-purpose data-cleaning rule.
    """
    if umbrella_file is None or not umbrella_file.exists():
        return df

    umbrella_df = pd.read_excel(umbrella_file, sheet_name="Renaming")
    if "Umbrella" not in umbrella_df.columns:
        raise ValueError("Merging_names.xlsx must contain an 'Umbrella' column.")

    umbrella_values = set(umbrella_df["Umbrella"].dropna().astype(str))
    filtered = df[
        df["group1_name"].astype(str).isin(umbrella_values)
        & df["group2_name"].astype(str).isin(umbrella_values)
    ].copy()
    print(f"Applied umbrella-name filtering: {len(filtered)} rows remain from {len(df)}.")
    return filtered


def run_pipeline(dataset: str, threshold: float, apply_umbrella_filter: bool) -> pd.DataFrame:
    """Run the MELTT pairwise transformation and return the final output table."""
    input_path = resolve_input_path(threshold)
    df = pd.read_csv(input_path, low_memory=False)

    if dataset == "meltt":
        df = df.copy()
    else:
        df = df[df["dataset"] == dataset].copy()

    if df.empty:
        print(f"Warning: no rows remain after filtering to dataset={dataset!r} for threshold={threshold}. Returning an empty output table.")
        empty_table = pd.DataFrame(columns=["group1_name", "group2_name", "year", "status", "Count", "source"])
        return empty_table

    df_fights = build_fights(df)
    df_cooperations = build_cooperations(df)
    df_output = pd.concat([df_fights, df_cooperations], ignore_index=True)
    df_output = df_output.sort_values(by=["group1_name", "group2_name", "year"]).reset_index(drop=True)
    df_output["source"] = "meltt"

    df_output = df_output[df_output["group1_name"] != df_output["group2_name"]].copy()

    if apply_umbrella_filter:
        umbrella_path = MERGING_DIR / "Merging_names.xlsx"
        df_output = filter_umbrella_matches(df_output, umbrella_path)

    return df_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct MELTT pairwise conflict and cooperation tables from the RF-expanded dataset."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=sorted(VALID_DATASETS),
        help="Dataset to use: 'meltt' = all rows, 'ucdp' or 'acled' = filter to one source. Default: meltt.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="RF threshold used to select the MELTT expanded source file. Default: 0.5.",
    )
    parser.add_argument(
        "--no-umbrella-filter",
        action="store_true",
        help="Disable the notebook's umbrella-name filtering. Enabled by default to preserve the original logic.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output CSV path. If omitted, the file is saved to the project pairwise-connections folder.",
    )
    args = parser.parse_args()

    output_df = run_pipeline(
        dataset=args.dataset,
        threshold=args.threshold,
        apply_umbrella_filter=not args.no_umbrella_filter,
    )

    if args.output_path is None:
        output_dir = PROJECT_ROOT / "01 - Data" / "03 - Pairwise_connections"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"meltt_pairwise_{args.dataset}_{args.threshold}.csv"
    else:
        output_path = args.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(output_path, index=False)
    print(f"Saved MELTT pairwise output to: {output_path}")
    print(output_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
