"""ACLED data cleaning pipeline.

This script loads the ACLED Syria event file, filters battle events, expands
semicolon-delimited actor fields, maps actor names to umbrella categories, and
exports a cleaned dataset aligned to the downstream UCDP-style structure.

Project paths are centralized in project_paths.py so the workflow can be
reproduced by setting the PROJECT_ROOT entry in .env once.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from project_paths import CLEANED_DIR, MERGING_DIR, SOURCE_DIR


def load_acled_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the raw ACLED event file and the umbrella-mapping tables used in the cleaning pipeline."""
    df_acled = pd.read_csv(
        SOURCE_DIR / "ACLED" / "2011-01-01-2025-02-04-Syria.csv"
    )

    df_merging1 = pd.read_excel(
        MERGING_DIR / "ACLED_merging_2rounds.xlsx",
        sheet_name="Original",
    )

    df_merging2 = pd.read_excel(
        MERGING_DIR / "ACLED_merging_2rounds.xlsx",
        sheet_name="Merging2_New",
    )

    return df_acled, df_merging1, df_merging2


def clean_acled(df_acled: pd.DataFrame, df_merging1: pd.DataFrame, df_merging2: pd.DataFrame) -> pd.DataFrame:
    """Standardize ACLED actor fields and align names with the downstream UCDP-compatible structure."""
    # Restrict to battle events to keep the dataset focused on the same conflict event type.
    df_acled = df_acled[df_acled["event_type"] == "Battles"].copy()

    # Split the semicolon-delimited actor strings into separate columns.
    actors1 = df_acled["assoc_actor_1"].str.split(";", expand=True)
    actors1.columns = [f"actor1_{i + 1}" for i in range(actors1.shape[1])]
    df_acled = pd.concat([df_acled, actors1], axis=1)

    actors2 = df_acled["assoc_actor_2"].str.split(";", expand=True)
    actors2.columns = [f"actor2_{i + 1}" for i in range(actors2.shape[1])]
    df_acled = pd.concat([df_acled, actors2], axis=1)

    # Round 1: map actor names to umbrella labels using the original manual matching table.
    actor_to_umbrella = dict(zip(df_merging1["actor"], df_merging1["Umbrella_name_1"]))
    for col in df_acled.filter(regex=r"^actor").columns:
        df_acled[col] = df_acled[col].str.strip()
        df_acled[col] = df_acled[col].map(actor_to_umbrella).fillna(df_acled[col])
        df_acled[col] = df_acled[col].apply(
            lambda x: "Military Forces of Syria"
            if isinstance(x, str)
            and (
                "Communal Militia (Syria)" in x
                or "Military Forces of Syria" in x
                or "Militia (Syria)" in x
                or "Police Forces of Syria (2000-)" in x
            )
            else x
        )

    # Round 2: apply the newer umbrella mapping table.
    actor_to_umbrella2 = dict(zip(df_merging2["Actor"], df_merging2["Umbrella"]))
    for col in df_acled.filter(regex=r"^actor").columns:
        df_acled[col] = df_acled[col].str.strip()
        df_acled[col] = df_acled[col].map(actor_to_umbrella2).fillna(df_acled[col])

    # Align ACLED column names to the UCDP-style naming convention used downstream.
    # ACLED does not provide a direct `source_article` column; the closest
    # provenance equivalents are the `source` and `notes` fields.
    acled_to_ucdp_mapping = {
        "event_id_cnty": "id",
        "year": "year",
        "latitude": "latitude",
        "longitude": "longitude",
        "fatalities": "best_est",
        "event_date": "date_start",
        "time_precision": "date_prec",
        "interaction": "violence_tax",
        "geo_precision": "where_prec",
        "source": "source_office",
        "admin1": "adm_1",
        "admin2": "adm_2",
        "location": "where_coordinates",
        "notes": "source_original",
        "actor1": "Side_a_1",
        "actor2": "Side_b_1",
    }

    for i in range(1, 12):
        acled_to_ucdp_mapping[f"actor1_{i}"] = f"Side_a_{i + 1}"
        acled_to_ucdp_mapping[f"actor2_{i}"] = f"Side_b_{i + 1}"

    df_acled_renamed = df_acled.rename(columns=acled_to_ucdp_mapping)
    df_acled_renamed["id"] = df_acled_renamed["id"].astype(str)
    df_acled_renamed["date_start"] = pd.to_datetime(df_acled_renamed["date_start"], errors="coerce")

    return df_acled_renamed


def main() -> None:
    """Execute the ACLED cleaning workflow and save the cleaned output file."""
    df_acled, df_merging1, df_merging2 = load_acled_data()
    df_cleaned = clean_acled(df_acled, df_merging1, df_merging2)

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CLEANED_DIR / "acled_cleaned.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"Saved ACLED cleaned data to: {output_path}")


if __name__ == "__main__":
    main()
