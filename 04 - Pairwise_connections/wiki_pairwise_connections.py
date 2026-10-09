"""Prepare the Conflict Wiki pairwise relationship dataset.

This script rebuilds the pairwise Conflict Wiki pipeline used in the notebook,
including the fight and cooperation construction steps, the entity-name mapping,
and the final output export.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import CLEANED_DIR, PROJECT_ROOT as REPO_ROOT, SOURCE_DIR

PROJECT_ROOT = REPO_ROOT

HISTORICAL_END_YEAR = 2022
UPDATE_START_YEAR = 2023
HISTORICAL_INPUT_PATH = CLEANED_DIR / "Wiki_cleaned.csv"
UPDATE_INPUT_PATH = CLEANED_DIR / "Wiki_cleaned_2324.csv"
ENTITY_PATH = SOURCE_DIR / "Conflict-Wiki" / "entity.csv"
NAMED_ENTITY_SOURCE_PATHS = (
    SOURCE_DIR / "Conflict-Wiki" / "wiki_ids_syria.csv",
    SOURCE_DIR / "Conflict-Wiki" / "pokus.csv",
)
PAIRWISE_DIR = PROJECT_ROOT / "01 - Data" / "03 - Pairwise_connections"


def load_inputs(
    historical_input_path: Path = HISTORICAL_INPUT_PATH,
    update_input_path: Path = UPDATE_INPUT_PATH,
    entity_path: Path = ENTITY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the historical Wiki data, the update, and the base entity table."""
    historical_df = pd.read_csv(historical_input_path)
    update_df = pd.read_csv(update_input_path)
    entity_df = pd.read_csv(entity_path, index_col=0)
    return historical_df, update_df, entity_df


def normalize_wiki_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Convert stringified list columns into actual Python lists."""
    df = df.copy()
    for col in ["Side_A", "Side_B", "Side_C", "Side_D"]:
        df[col] = df[col].apply(ast.literal_eval)
    df["year"] = df["year"].apply(ast.literal_eval)
    return df


def restrict_years(
    df: pd.DataFrame,
    minimum_year: int | None = None,
    maximum_year: int | None = None,
) -> pd.DataFrame:
    """Restrict every conflict's active-year list to a non-overlapping window."""
    df = df.copy()

    def keep_years(years: list[int]) -> list[int]:
        return [
            int(year)
            for year in years
            if (minimum_year is None or int(year) >= minimum_year)
            and (maximum_year is None or int(year) <= maximum_year)
        ]

    df["year"] = df["year"].apply(keep_years)
    return df[df["year"].map(bool)].copy()


def load_supplemental_entity_names(
    source_paths: Iterable[Path],
) -> tuple[pd.DataFrame, dict[int, set[str]]]:
    """Recover positive Wikipedia page-ID/name pairs retained by the scraper.

    The 2025 scrape stored newer page IDs that are absent from the older
    Conflict-Wiki entity table. Its named intermediate CSVs retain the matching
    titles. Negative IDs are deliberately excluded because the same negative
    placeholder was reused for many unrelated unresolved titles.
    """
    id_to_names: dict[int, set[str]] = defaultdict(set)

    for source_path in source_paths:
        if not source_path.exists():
            continue
        source_df = pd.read_csv(source_path, low_memory=False)
        actor_columns = [
            column
            for column in source_df.columns
            if column.startswith("combatant") or column.startswith("units")
        ]
        for column in actor_columns:
            for value in source_df[column].dropna():
                try:
                    entries = ast.literal_eval(value) if isinstance(value, str) else value
                except (SyntaxError, ValueError):
                    continue
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                        continue
                    name, entity_id = entry[0], entry[1]
                    try:
                        entity_id = int(entity_id)
                    except (TypeError, ValueError):
                        continue
                    if entity_id <= 0 or not isinstance(name, str) or not name.strip():
                        continue
                    id_to_names[entity_id].add(name.strip())

    ambiguous = {
        entity_id: sorted(names)
        for entity_id, names in id_to_names.items()
        if len(names) > 1
    }
    if ambiguous:
        examples = list(ambiguous.items())[:5]
        raise ValueError(
            "Positive Wikipedia entity IDs mapped to multiple names in the "
            f"named scrape sources. Examples: {examples}"
        )

    supplemental = pd.DataFrame(
        [
            {"entity_id": entity_id, "entity_name": next(iter(names))}
            for entity_id, names in sorted(id_to_names.items())
        ]
    )
    return supplemental, id_to_names


def augment_entity_names(
    entity_df: pd.DataFrame,
    supplemental_df: pd.DataFrame,
) -> pd.DataFrame:
    """Append newer positive page IDs without overriding historical names."""
    base = entity_df[["entity_id", "entity_name"]].copy()
    base["entity_id"] = pd.to_numeric(base["entity_id"], errors="raise").astype(int)
    if supplemental_df.empty:
        return base

    supplemental_df = supplemental_df.copy()
    supplemental_df["entity_id"] = pd.to_numeric(
        supplemental_df["entity_id"], errors="raise"
    ).astype(int)
    supplemental_df = supplemental_df[
        ~supplemental_df["entity_id"].isin(base["entity_id"])
    ]
    return pd.concat([base, supplemental_df], ignore_index=True).drop_duplicates(
        subset=["entity_id"], keep="first"
    )


def build_fights(df: pd.DataFrame) -> pd.DataFrame:
    """Construct enemy dyads from all pairwise combinations across sides."""
    df_exploded = df.explode("Side_A").explode("Side_B").explode("Side_C").explode("Side_D").explode("year")
    df_exploded = df_exploded.reset_index(drop=True)

    cooperation_columns = [col for col in df_exploded.columns if col.startswith("Side_")]
    df_fighting = df_exploded.melt(
        id_vars=["conflict_id", "year"],
        value_vars=cooperation_columns,
        var_name="side_column",
        value_name="Fights",
    )
    df_fighting = df_fighting.drop_duplicates()
    df_fighting = df_fighting[df_fighting["Fights"].astype(bool)].copy()

    grouped = df_fighting.groupby(["conflict_id", "year"])

    def create_pairs(group: pd.DataFrame) -> list[tuple[int, int]] | None:
        sides_dict: dict[str, list] = {}
        for side in ["Side_A", "Side_B", "Side_C", "Side_D"]:
            sides_dict[side] = group[group["side_column"] == side]["Fights"].tolist()

        pairs: list[tuple[int, int]] = []
        sides = ["Side_A", "Side_B", "Side_C", "Side_D"]
        for i in range(len(sides)):
            for j in range(i + 1, len(sides)):
                side1, side2 = sides[i], sides[j]
                for fight1 in sides_dict[side1]:
                    for fight2 in sides_dict[side2]:
                        pairs.append((fight1, fight2))
        return pairs if pairs else None

    df_pairs = (
        grouped.apply(create_pairs, include_groups=False)
        .explode()
        .dropna()
        .apply(pd.Series)
        .rename(columns={0: "group1_id", 1: "group2_id"})
        .reset_index()
    )

    df_pairs[["group1_sorted", "group2_sorted"]] = df_pairs.apply(
        lambda row: pd.Series(sorted([row["group1_id"], row["group2_id"]])),
        axis=1,
    )
    df_pairs = df_pairs[df_pairs["group1_sorted"] != df_pairs["group2_sorted"]].copy()

    df_fights = (
        df_pairs.groupby(["group1_sorted", "group2_sorted", "year"])
        .size()
        .reset_index(name="Count")
    )
    df_fights["source"] = "Wiki"
    df_fights["status"] = "Enemies"
    return df_fights


def generate_cooperation_pairs(df: pd.DataFrame, side_column: str) -> pd.DataFrame:
    """Construct cooperation dyads for a specific side in each event-year."""
    df_side = df[["conflict_id", side_column, "year"]].explode("year")
    side_expanded = pd.DataFrame(df_side[side_column].tolist(), index=df_side.index).add_prefix("Cooperation_")
    df_side = pd.concat([df_side, side_expanded], axis=1)

    cooperations_columns = [col for col in df_side.columns if col.startswith("Cooperation_")]
    df_long = df_side.melt(
        id_vars=["conflict_id", "year"],
        value_vars=cooperations_columns,
        var_name=f"{side_column}_column",
        value_name="Cooperation",
    )
    df_long = df_long.dropna(subset=["Cooperation"]).copy()
    df_long = df_long.drop_duplicates().copy()

    df_pairs = (
        df_long.groupby(["conflict_id", "year"])["Cooperation"]
        .apply(lambda x: list(combinations(pd.unique(x), 2)))
        .explode()
        .dropna()
        .apply(pd.Series)
        .rename(columns={0: "Ally1_id", 1: "Ally2_id"})
        .reset_index()
    )

    df_pairs = df_pairs[df_pairs["Ally1_id"] != df_pairs["Ally2_id"]].copy()
    df_pairs[["group1_sorted", "group2_sorted"]] = df_pairs.apply(
        lambda row: pd.Series(sorted([row["Ally1_id"], row["Ally2_id"]])),
        axis=1,
    )
    return df_pairs


def build_cooperations(
    df: pd.DataFrame,
    cooperation_sides: tuple[str, ...] = ("Side_A", "Side_B", "Side_C"),
) -> pd.DataFrame:
    """Combine cooperation dyads using the sides included by the past workflow.

    The Dropbox Wiki notebooks used Side A, B, and C for cooperation pairs.
    Side D remained part of the cross-side fight construction.
    """
    pair_frames = [
        generate_cooperation_pairs(df, side_column)
        for side_column in cooperation_sides
    ]
    df_cooperations = pd.concat(pair_frames, ignore_index=True)

    df_cooperations = (
        df_cooperations.groupby(["group1_sorted", "group2_sorted", "year"])
        .size()
        .reset_index(name="Count")
    )
    df_cooperations["source"] = "Wiki"
    df_cooperations["status"] = "Allies"
    return df_cooperations


def map_entity_names(
    df_final: pd.DataFrame,
    entity_df: pd.DataFrame,
    segment: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map entity IDs and return both mapped rows and an unmapped-ID audit."""
    name_lookup = (
        entity_df[["entity_id", "entity_name"]]
        .dropna(subset=["entity_id", "entity_name"])
        .drop_duplicates(subset=["entity_id"], keep="first")
        .set_index("entity_id")["entity_name"]
    )

    df_final = df_final.copy()
    df_final["group1_name"] = df_final["group1_sorted"].map(name_lookup)
    df_final["group2_name"] = df_final["group2_sorted"].map(name_lookup)

    unresolved_frames = []
    for id_column, name_column in (
        ("group1_sorted", "group1_name"),
        ("group2_sorted", "group2_name"),
    ):
        unresolved = df_final.loc[df_final[name_column].isna(), [id_column]].copy()
        unresolved = unresolved.rename(columns={id_column: "entity_id"})
        unresolved_frames.append(unresolved)

    if unresolved_frames:
        unresolved_ids = pd.concat(unresolved_frames, ignore_index=True)
        unmapped_audit = (
            unresolved_ids.groupby("entity_id")
            .size()
            .reset_index(name="pairwise_row_occurrences")
        )
        unmapped_audit.insert(0, "segment", segment)
    else:
        unmapped_audit = pd.DataFrame(
            columns=["segment", "entity_id", "pairwise_row_occurrences"]
        )

    mapped = df_final.dropna(subset=["group1_name", "group2_name"]).copy()
    return mapped, unmapped_audit


def build_output(
    df: pd.DataFrame,
    entity_df: pd.DataFrame,
    segment: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Build, map, and audit one non-overlapping Wiki time segment."""
    df_fights = build_fights(df)
    df_cooperations = build_cooperations(df)
    df_final = pd.concat([df_cooperations, df_fights], ignore_index=True)
    rows_before_mapping = len(df_final)

    df_final["group1_sorted"] = df_final["group1_sorted"].astype(int)
    df_final["group2_sorted"] = df_final["group2_sorted"].astype(int)
    df_final, unmapped_audit = map_entity_names(
        df_final,
        entity_df,
        segment=segment,
    )

    output_df = df_final[["group1_name", "group2_name", "year", "status", "source", "Count"]].copy()
    output_df = output_df[output_df["group1_name"] != output_df["group2_name"]].copy()
    output_df = output_df.sort_values(
        ["group1_name", "group2_name", "year", "status"]
    ).reset_index(drop=True)
    return output_df, unmapped_audit, rows_before_mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one Wiki pairwise dataset from the historical data through "
            "2022 and the separately maintained 2023-onward update."
        )
    )
    parser.add_argument(
        "--historical-input-path",
        type=Path,
        default=HISTORICAL_INPUT_PATH,
        help="Historical cleaned Wiki source used through 2022.",
    )
    parser.add_argument(
        "--update-input-path",
        type=Path,
        default=UPDATE_INPUT_PATH,
        help="Cleaned Wiki update used from 2023 onward.",
    )
    parser.add_argument(
        "--entity-path",
        type=Path,
        default=ENTITY_PATH,
        help="Historical Conflict-Wiki entity mapping file.",
    )
    parser.add_argument(
        "--named-entity-source-paths",
        type=Path,
        nargs="*",
        default=list(NAMED_ENTITY_SOURCE_PATHS),
        help=(
            "Named scraper intermediates used to supplement newer positive "
            "Wikipedia page IDs."
        ),
    )
    parser.add_argument(
        "--historical-output-path",
        type=Path,
        default=PAIRWISE_DIR / "ConflictWiki-connections_withcounts_historical_through_2022.csv",
        help="Audit output for the frozen historical segment.",
    )
    parser.add_argument(
        "--update-output-path",
        type=Path,
        default=PAIRWISE_DIR / "ConflictWiki-connections_withcounts_2023_onward.csv",
        help="Audit output for the non-overlapping update segment.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PAIRWISE_DIR / "ConflictWiki-connections_withcounts_new.csv",
        help=(
            "Combined output path. The default matches the Wiki filename read "
            "by the Dropbox Dyadic - MAIN notebook."
        ),
    )
    parser.add_argument(
        "--compatibility-output-path",
        type=Path,
        default=PAIRWISE_DIR / "ConflictWiki-connections_withcounts.csv",
        help="Compatibility copy for the newer repository filename.",
    )
    parser.add_argument(
        "--unmapped-output-path",
        type=Path,
        default=PAIRWISE_DIR / "ConflictWiki-unmapped-entity-ids.csv",
        help="Audit table of entity IDs that could not be mapped to names.",
    )
    args = parser.parse_args()

    historical_df, update_df, entity_df = load_inputs(
        historical_input_path=args.historical_input_path,
        update_input_path=args.update_input_path,
        entity_path=args.entity_path,
    )
    supplemental_df, _ = load_supplemental_entity_names(
        args.named_entity_source_paths
    )
    entity_df = augment_entity_names(entity_df, supplemental_df)

    historical_df = restrict_years(
        normalize_wiki_lists(historical_df),
        maximum_year=HISTORICAL_END_YEAR,
    )
    update_df = restrict_years(
        normalize_wiki_lists(update_df),
        minimum_year=UPDATE_START_YEAR,
    )

    historical_output, historical_unmapped, historical_pre_mapping = build_output(
        historical_df,
        entity_df,
        segment="historical_through_2022",
    )
    update_output, update_unmapped, update_pre_mapping = build_output(
        update_df,
        entity_df,
        segment="update_2023_onward",
    )

    if not historical_output.empty and historical_output["year"].max() > HISTORICAL_END_YEAR:
        raise ValueError("Historical Wiki output contains a year after 2022.")
    if not update_output.empty and update_output["year"].min() < UPDATE_START_YEAR:
        raise ValueError("Wiki update contains a year before 2023.")

    output_df = pd.concat(
        [historical_output, update_output],
        ignore_index=True,
    ).sort_values(
        ["group1_name", "group2_name", "year", "status", "source"]
    ).reset_index(drop=True)
    output_df = output_df.drop_duplicates().reset_index(drop=True)
    unmapped_output = pd.concat(
        [historical_unmapped, update_unmapped],
        ignore_index=True,
    ).sort_values(["segment", "pairwise_row_occurrences", "entity_id"], ascending=[True, False, True])

    for output_path in (
        args.historical_output_path,
        args.update_output_path,
        args.output_path,
        args.compatibility_output_path,
        args.unmapped_output_path,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    historical_output.to_csv(args.historical_output_path, index=False)
    update_output.to_csv(args.update_output_path, index=False)
    output_df.to_csv(args.output_path, index=False)
    output_df.to_csv(args.compatibility_output_path, index=False)
    unmapped_output.to_csv(args.unmapped_output_path, index=False)

    print(
        "Historical through 2022: "
        f"{historical_pre_mapping} rows before name mapping, "
        f"{len(historical_output)} rows saved."
    )
    print(
        "Update from 2023 onward: "
        f"{update_pre_mapping} rows before name mapping, "
        f"{len(update_output)} rows saved."
    )
    print(f"Saved historical segment to: {args.historical_output_path}")
    print(f"Saved update segment to: {args.update_output_path}")
    print(f"Saved combined Dropbox-compatible output to: {args.output_path}")
    print(f"Saved repository compatibility copy to: {args.compatibility_output_path}")
    print(f"Saved unmapped entity audit to: {args.unmapped_output_path}")
    print(f"Combined rows: {len(output_df)}")


if __name__ == "__main__":
    main()
