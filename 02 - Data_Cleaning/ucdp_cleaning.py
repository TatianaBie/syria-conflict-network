"""UCDP data cleaning pipeline.

This script loads the raw UCDP event file, standardizes actor labels, expands side
names into structured columns, maps them to umbrella categories, and exports a
cleaned dataset for downstream merging and analysis.

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


def load_ucdp_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw UCDP event file and the umbrella-mapping table used in the cleaning pipeline."""
    df_matching = pd.read_excel(
        MERGING_DIR / "Merging_groups_UCDP.xlsx",
        sheet_name="MAIN_merging",
    )

    cols = [
        "id",
        "relid",
        "year",
        "active_year",
        "code_status",
        "type_of_violence",
        "conflict_dset_id",
        "conflict_new_id",
        "conflict_name",
        "dyad_dset_id",
        "dyad_new_id",
        "dyad_name",
        "side_a_dset_id",
        "side_a_new_id",
        "side_a",
        "side_b_dset_id",
        "side_b_new_id",
        "side_b",
        "number_of_sources",
        "source_article",
        "source_office",
        "source_date",
        "source_headline",
        "source_original",
        "where_prec",
        "where_coordinates",
        "where_description",
        "adm_1",
        "adm_2",
        "latitude",
        "longitude",
        "geom_wkt",
        "priogrid_gid",
        "country",
        "country_id",
        "region",
        "event_clarity",
        "date_prec",
        "date_start",
        "date_end",
        "deaths_a",
        "deaths_b",
        "deaths_civilians",
        "deaths_unknown",
        "best_est",
        "high_est",
        "low_est",
    ]

    df_ucdp = pd.read_csv(
        SOURCE_DIR / "UCDP" / "gedevents-2026-02-06.csv",
        names=cols,
        header=0,
        encoding="utf-8-sig",
        index_col="id",
        usecols=range(len(cols)),
    )

    return df_ucdp, df_matching


def clean_ucdp(df_ucdp: pd.DataFrame, df_matching: pd.DataFrame) -> pd.DataFrame:
    """Standardize side names, expand them into actor columns, and map raw labels to umbrella categories."""
    replacements = {
        "Mare’ Operations Room": "Sham Legion, Free Syrian Army, The Levantine Front",
        "Euphrates Vulcano": "Peoples Protection Units, Free Syrian Army, Liwa Thuwar ar-Raqqa",
        "Fatah Halab": "The Levantine Front, HTS: Hayat Tahrir al-Sham, Sham Legion",
        "Islamic Front": "Jaysh al-Islam, Ahrar al-Sham, Liwa al-Tawhid",
        "Bayareq al-Shaaitat ": "Bayareq al-Shaaitat",
        "Army of Ahl al-Sunni wal Jamaa ": "Army of Ahl al-Sunni wal Jamaa",
    }

    # Apply the manual normalization entries before splitting the actor strings.
    for column in ["side_a", "side_b"]:
        df_ucdp[column] = df_ucdp[column].astype(str)
        for old_value, new_value in replacements.items():
            df_ucdp[column] = df_ucdp[column].str.replace(old_value, new_value)

    # Expand the side names into separate columns for the UCDP-style structure.
    df_split_a = df_ucdp["side_a"].str.split(", ", expand=True)
    df_split_a.columns = [f"Side_a_{i + 1}" for i in range(df_split_a.shape[1])]
    df_ucdp = pd.concat([df_ucdp, df_split_a], axis=1)

    df_split_b = df_ucdp["side_b"].str.split(", ", expand=True)
    df_split_b.columns = [f"Side_b_{i + 1}" for i in range(df_split_b.shape[1])]
    df_ucdp = pd.concat([df_ucdp, df_split_b], axis=1)

    # Replace raw names with umbrella labels from the mapping file.
    mapping_dict = dict(zip(df_matching["Name"], df_matching["Umbrella"]))
    for col in df_ucdp.columns:
        if col.startswith("Side_a_") or col.startswith("Side_b_"):
            df_ucdp[col] = df_ucdp[col].replace(mapping_dict)

    df_ucdp["date_start"] = pd.to_datetime(df_ucdp["date_start"], errors="coerce")
    df_ucdp["date_end"] = pd.to_datetime(df_ucdp["date_end"], errors="coerce")

    return df_ucdp


def main() -> None:
    """Execute the UCDP cleaning workflow and save the cleaned output file."""
    df_ucdp, df_matching = load_ucdp_data()
    df_cleaned = clean_ucdp(df_ucdp, df_matching)

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CLEANED_DIR / "ucdp_cleaned.csv"
    df_cleaned.to_csv(output_path)
    print(f"Saved UCDP cleaned data to: {output_path}")


if __name__ == "__main__":
    main()
