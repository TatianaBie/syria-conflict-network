#!/usr/bin/env python3
"""Run the complete period and yearly territorial-presence workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

from estimate_territorial_presence import (
    ACTORS,
    IS_NAME,
    PERIODS,
    aggregate_shares,
    estimate_period,
    integrate_cell_population,
    load_actor_events,
    load_actor_mapping,
    load_grid,
)

DEFAULT_THRESHOLD = 0.5


def threshold_label(value: float) -> str:
    """Format a threshold the same way as the monadic/dyadic filenames."""
    return f"{value:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        help=(
            "Optional explicit event CSV. By default, resolve "
            "dataset_threshold_{threshold}_meltt.csv from the canonical "
            "repository source-data directory."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="MELTT RF threshold used by the monadic/dyadic pipeline (default: 0.5).",
    )
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        help=(
            "Directory containing dataset_threshold_{threshold}_meltt.csv. "
            "Defaults to 01 - Data/02 - Source_Data-cleaned in the repository."
        ),
    )
    parser.add_argument("--actor-mapping", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--population-dir", type=Path, required=True)
    parser.add_argument("--initial-period-shares", type=Path, required=True)
    parser.add_argument(
        "--previous-period-shares",
        type=Path,
        help="Optional period shares from the previous population-weighting method.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bandwidth-km", type=float, default=20.0)
    parser.add_argument("--cutoff-sigma", type=float, default=3.0)
    parser.add_argument("--prior", type=float, default=0.05)
    return parser.parse_args()


def resolve_events_path(args: argparse.Namespace) -> Path:
    """Resolve the canonical thresholded MELTT input unless explicitly overridden."""
    if args.events is not None:
        path = args.events
    else:
        repository_root = Path(__file__).resolve().parents[2]
        source_dir = (
            args.source_data_dir
            if args.source_data_dir is not None
            else repository_root / "01 - Data" / "02 - Source_Data-cleaned"
        )
        path = source_dir / f"dataset_threshold_{threshold_label(args.threshold)}_meltt.csv"
    if not path.exists():
        raise FileNotFoundError(f"Canonical MELTT event input not found: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conditional_non_is(shares: pd.DataFrame, period_column: str) -> pd.DataFrame:
    columns = [actor for actor in ACTORS if actor != IS_NAME]
    result = shares[[period_column, *columns]].copy()
    result[columns] = result[columns].div(result[columns].sum(axis=1), axis=0)
    return result


def shares_by_actor(shares: pd.DataFrame, period_column: str) -> pd.DataFrame:
    """Return the same shares with actors in rows and periods in columns."""
    return (
        shares.set_index(period_column)
        .reindex(columns=ACTORS)
        .T.rename_axis("umbrella_actor")
        .reset_index()
    )


def _write_frame(
    sheet,
    frame: pd.DataFrame,
    *,
    start_row: int = 1,
    banded_rows: bool = True,
) -> tuple[int, int]:
    """Write one dataframe and apply the shared review-workbook style."""
    clean = frame.astype(object).where(pd.notna(frame), None)
    rows = list(dataframe_to_rows(clean, index=False, header=True))
    for row_offset, values in enumerate(rows):
        for column, value in enumerate(values, start=1):
            sheet.cell(start_row + row_offset, column, value)

    last_row = start_row + len(rows) - 1
    last_column = len(frame.columns)
    header_fill = PatternFill("solid", fgColor="193B63")
    alternate_fill = PatternFill("solid", fgColor="D9EEF7")
    for cell in sheet[start_row][:last_column]:
        cell.fill = header_fill
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.row_dimensions[start_row].height = 30

    for row in range(start_row + 1, last_row + 1):
        if banded_rows and (row - start_row) % 2 == 1:
            for cell in sheet[row][:last_column]:
                cell.fill = alternate_fill
        for cell in sheet[row][:last_column]:
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="center")

    for column, name in enumerate(frame.columns, start=1):
        values = [str(name), *["" if value is None else str(value) for value in clean[name]]]
        width = min(max(len(value) for value in values) + 2, 38)
        sheet.column_dimensions[get_column_letter(column)].width = max(width, 11)
    sheet.sheet_view.showGridLines = False
    return last_row, last_column


def save_review_workbook(
    path: Path,
    threshold: float,
    period_shares_by_actor: pd.DataFrame,
    period_shares: pd.DataFrame,
    yearly_shares: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
    manifest: pd.DataFrame,
    previous_comparison: pd.DataFrame | None = None,
    previous_summary: pd.DataFrame | None = None,
) -> None:
    """Create the formatted workbook from the same dataframes as the CSV outputs."""
    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "Summary"

    summary_sheet["A2"] = "Territorial-presence rerun"
    summary_sheet["A2"].font = Font(name="Arial", size=14, bold=True, color="193B63")
    summary_sheet["A4"] = (
        "Full event-level rerun from the canonical MELTT "
        f"threshold-{threshold_label(threshold)} input, with {len(ACTORS)} "
        "umbrella actors, WorldPop weighting, and monadic-style "
        "RF-row deduplication."
    )
    summary_sheet["A4"].font = Font(name="Arial", size=10, italic=True, color="4A4A4A")

    share_sums = period_shares.set_index("period")[list(ACTORS)].sum(axis=1)
    summary_table = summary.copy()
    summary_table["rerun_share_sum"] = summary_table["period"].map(share_sums)
    summary_table = summary_table.rename(
        columns={
            "period": "Period",
            "total_variation_percentage_points": "Distribution shift (pp)",
            "largest_absolute_change_actor": "Largest-change actor",
            "largest_absolute_change_percentage_points": "Largest change (pp)",
            "initial_top_actor": "Initial top actor",
            "rerun_top_actor": "Rerun top actor",
            "initial_turkey_share": "Initial Turkey",
            "rerun_turkey_share": "Rerun Turkey",
            "initial_is_share": "Initial IS",
            "rerun_is_share": "Rerun IS",
            "pearson_correlation": "Correlation",
            "rerun_share_sum": "Rerun share sum",
        }
    )
    _write_frame(summary_sheet, summary_table, start_row=6, banded_rows=False)
    for row in range(7, 10):
        for column in (2, 4, 11):
            summary_sheet.cell(row, column).number_format = "0.000"
        for column in (7, 8, 9, 10, 12):
            summary_sheet.cell(row, column).number_format = "0.000%"

    method_rows = [
        ("Method", ""),
        (
            "Actor universe",
            f"{len(ACTORS)} actors; Operation Peace Spring maps to Turkey",
        ),
        ("Spatial model", "20 km Gaussian bandwidth; kernels truncated at 60 km"),
        (
            "Population weighting",
            "Events weighted equally; WorldPop summed within cells at aggregation",
        ),
        ("Period rasters", "2013, 2017 and 2020 for the three four-year windows"),
        ("Yearly rasters", "Same-year raster through 2020; 2020 raster for 2021–2024"),
        (
            "Interpretation",
            "Population-weighted spatial conflict presence, not independently observed territorial control",
        ),
    ]
    for row, (label, value) in enumerate(method_rows, start=11):
        summary_sheet.cell(row, 1, label)
        summary_sheet.cell(row, 2, value)
        summary_sheet.cell(row, 1).font = Font(name="Arial", size=10, bold=True)
        summary_sheet.cell(row, 2).font = Font(name="Arial", size=10)
    for cell in summary_sheet[11][:12]:
        cell.fill = PatternFill("solid", fgColor="D9EAF7")
        cell.font = Font(name="Arial", size=10, bold=True, color="193B63")
    summary_sheet.column_dimensions["A"].width = 22
    summary_sheet.column_dimensions["B"].width = 38
    summary_sheet.freeze_panes = "A6"

    period_sheet = workbook.create_sheet("Period shares")
    period_last_row, period_last_column = _write_frame(
        period_sheet, period_shares_by_actor
    )
    for row in period_sheet.iter_rows(
        min_row=2, max_row=period_last_row, min_col=2, max_col=period_last_column
    ):
        for cell in row:
            cell.number_format = "0.000%"
    period_sheet.freeze_panes = "B2"

    yearly_sheet = workbook.create_sheet("Yearly shares")
    yearly_last_row, yearly_last_column = _write_frame(yearly_sheet, yearly_shares)
    for row in yearly_sheet.iter_rows(
        min_row=2, max_row=yearly_last_row, min_col=2, max_col=yearly_last_column
    ):
        for cell in row:
            cell.number_format = "0.000%"
    yearly_sheet.freeze_panes = "B2"

    comparison_sheet = workbook.create_sheet("Actor comparison")
    comparison_last_row, _ = _write_frame(comparison_sheet, comparison)
    for row in comparison_sheet.iter_rows(
        min_row=2, max_row=comparison_last_row, min_col=3, max_col=5
    ):
        for cell in row:
            cell.number_format = "0.000%"
    for cell in comparison_sheet["F"][1:comparison_last_row]:
        cell.number_format = "0.000"
    comparison_sheet.freeze_panes = "C2"

    if previous_comparison is not None and previous_summary is not None:
        method_sheet = workbook.create_sheet("Population method comparison")
        method_summary = previous_summary.rename(
            columns={
                "period": "Period",
                "total_variation_percentage_points": "Distribution shift (pp)",
                "largest_absolute_change_actor": "Largest-change actor",
                "largest_absolute_change_percentage_points": "Largest change (pp)",
                "previous_top_actor": "Previous top actor",
                "updated_top_actor": "Updated top actor",
                "previous_turkey_share": "Previous Turkey",
                "updated_turkey_share": "Updated Turkey",
                "previous_is_share": "Previous IS",
                "updated_is_share": "Updated IS",
                "pearson_correlation": "Correlation",
            }
        )
        summary_last_row, _ = _write_frame(
            method_sheet, method_summary, banded_rows=False
        )
        for row in range(2, summary_last_row + 1):
            for column in (2, 4, 11):
                method_sheet.cell(row, column).number_format = "0.000"
            for column in (7, 8, 9, 10):
                method_sheet.cell(row, column).number_format = "0.000%"

        detail_start = summary_last_row + 3
        detail_last_row, _ = _write_frame(
            method_sheet,
            previous_comparison,
            start_row=detail_start,
        )
        for row in method_sheet.iter_rows(
            min_row=detail_start + 1,
            max_row=detail_last_row,
            min_col=3,
            max_col=5,
        ):
            for cell in row:
                cell.number_format = "0.000%"
        for cell in method_sheet["F"][detail_start:detail_last_row]:
            cell.number_format = "0.000"
        method_sheet.freeze_panes = f"C{detail_start + 1}"

    audit_sheet = workbook.create_sheet("Mapping audit")
    _write_frame(audit_sheet, audit)
    audit_sheet.freeze_panes = "A2"

    inputs_sheet = workbook.create_sheet("Inputs")
    _write_frame(inputs_sheet, manifest.drop(columns="path"))
    inputs_sheet.column_dimensions["C"].width = 68
    inputs_sheet.freeze_panes = "A2"

    workbook.save(path)


def save_share_plot(shares: pd.DataFrame, index_column: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    top = shares[list(ACTORS)].mean(axis=0).nlargest(12).index.tolist()
    plot = shares.set_index(index_column)[top].copy()
    plot["Other actors"] = 1 - plot.sum(axis=1)
    width = max(10, len(plot) * 0.7)
    ax = plot.plot(kind="bar", stacked=True, figsize=(width, 6), width=0.78)
    ax.set(
        ylabel="Population-weighted spatial presence share",
        xlabel=index_column.capitalize(),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def comparison_with_initial(
    initial_path: Path, new_shares: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    initial = pd.read_csv(initial_path).set_index("period")
    new = new_shares.set_index("period")
    actors = sorted(set(initial.columns) | set(new.columns))
    initial = initial.reindex(columns=actors, fill_value=0.0)
    new = new.reindex(columns=actors, fill_value=0.0)

    records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for period in new.index:
        old_row = initial.loc[period]
        new_row = new.loc[period]
        difference = new_row - old_row
        for actor in actors:
            records.append(
                {
                    "period": period,
                    "actor": actor,
                    "initial_share": old_row[actor],
                    "rerun_share": new_row[actor],
                    "change_in_share": difference[actor],
                    "change_percentage_points": 100 * difference[actor],
                }
            )
        summaries.append(
            {
                "period": period,
                "total_variation_percentage_points": 50
                * difference.abs().sum(),
                "largest_absolute_change_actor": difference.abs().idxmax(),
                "largest_absolute_change_percentage_points": 100
                * difference.abs().max(),
                "initial_top_actor": old_row.idxmax(),
                "rerun_top_actor": new_row.idxmax(),
                "initial_turkey_share": old_row.get("Turkey", np.nan),
                "rerun_turkey_share": new_row.get("Turkey", np.nan),
                "initial_is_share": old_row.get(IS_NAME, np.nan),
                "rerun_is_share": new_row.get(IS_NAME, np.nan),
                "pearson_correlation": old_row.corr(new_row),
            }
        )
    return pd.DataFrame(records), pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = resolve_events_path(args)

    mapping, mapping_table = load_actor_mapping(args.actor_mapping)
    events, audit = load_actor_events(
        events_path,
        actor_mapping=mapping,
        deduplicate=True,
        start_year=2011,
        end_year=2024,
    )
    grid = load_grid(args.grid)
    grid.to_csv(args.output_dir / "grid_syria_0p1deg.csv", index=False)
    raster_years = sorted(
        {period[3] for period in PERIODS} | set(range(2011, 2021))
    )
    cell_populations = {
        year: integrate_cell_population(
            args.population_dir / f"syr_ppp_{year}.tif",
            grid.lon.to_numpy(),
            grid.lat.to_numpy(),
        )
        for year in raster_years
    }

    common = dict(
        bandwidth_km=args.bandwidth_km,
        prior_weight=args.prior,
        cutoff_sigma=args.cutoff_sigma,
    )
    period_frames = [
        estimate_period(
            events,
            grid,
            args.population_dir,
            period=period,
            cell_population=cell_populations[period[3]],
            **common,
        )
        for period in PERIODS
    ]
    period_allocations = pd.concat(period_frames, ignore_index=True)
    period_shares = aggregate_shares(period_allocations)

    yearly_periods = [
        (str(year), year, year, min(year, 2020)) for year in range(2011, 2025)
    ]
    yearly_frames = [
        estimate_period(
            events,
            grid,
            args.population_dir,
            period=period,
            cell_population=cell_populations[period[3]],
            **common,
        )
        for period in yearly_periods
    ]
    yearly_allocations = pd.concat(yearly_frames, ignore_index=True)
    yearly_shares = aggregate_shares(yearly_allocations).rename(
        columns={"period": "year"}
    )
    yearly_shares["year"] = yearly_shares["year"].astype(int)

    period_allocations.to_csv(
        args.output_dir / "spatial_actor_allocations_by_period.csv", index=False
    )
    yearly_allocations = yearly_allocations.rename(columns={"period": "year"})
    yearly_allocations["year"] = yearly_allocations["year"].astype(int)
    yearly_allocations.to_csv(
        args.output_dir / "spatial_actor_allocations_yearly.csv", index=False
    )
    period_shares.to_csv(args.output_dir / "actor_shares_by_period.csv", index=False)
    period_shares_by_actor = shares_by_actor(period_shares, "period")
    period_shares_by_actor.to_csv(
        args.output_dir / "actor_shares_by_period_by_actor.csv", index=False
    )
    yearly_shares.to_csv(args.output_dir / "actor_shares_yearly.csv", index=False)
    conditional_non_is(period_shares, "period").to_csv(
        args.output_dir / "actor_shares_by_period_conditional_on_non_IS.csv",
        index=False,
    )
    conditional_non_is(yearly_shares, "year").to_csv(
        args.output_dir / "actor_shares_yearly_conditional_on_non_IS.csv",
        index=False,
    )
    audit.to_csv(args.output_dir / "actor_mapping_audit.csv", index=False)
    (
        events.groupby(["year", "actor"])
        .size()
        .rename("actor_event_occurrences")
        .reset_index()
        .to_csv(args.output_dir / "actor_event_counts_by_year.csv", index=False)
    )

    comparison, summary = comparison_with_initial(
        args.initial_period_shares, period_shares
    )
    comparison.to_csv(
        args.output_dir / "actor_share_comparison_with_initial.csv", index=False
    )
    summary.to_csv(
        args.output_dir / "period_comparison_with_initial.csv", index=False
    )
    previous_comparison = None
    previous_summary = None
    if args.previous_period_shares is not None:
        previous_comparison, previous_summary = comparison_with_initial(
            args.previous_period_shares, period_shares
        )
        previous_comparison = previous_comparison.rename(
            columns={
                "initial_share": "previous_share",
                "rerun_share": "updated_share",
            }
        )
        previous_summary = previous_summary.rename(
            columns={
                "initial_top_actor": "previous_top_actor",
                "rerun_top_actor": "updated_top_actor",
                "initial_turkey_share": "previous_turkey_share",
                "rerun_turkey_share": "updated_turkey_share",
                "initial_is_share": "previous_is_share",
                "rerun_is_share": "updated_is_share",
            }
        )
        previous_comparison.to_csv(
            args.output_dir
            / "actor_share_comparison_with_previous_population_method.csv",
            index=False,
        )
        previous_summary.to_csv(
            args.output_dir
            / "period_comparison_with_previous_population_method.csv",
            index=False,
        )

    save_share_plot(
        period_shares,
        "period",
        args.output_dir / "actor_shares_by_period.png",
    )
    save_share_plot(
        yearly_shares,
        "year",
        args.output_dir / "actor_shares_yearly.png",
    )

    raster_paths = [
        args.population_dir / f"syr_ppp_{year}.tif" for year in range(2011, 2021)
    ]
    manifest_paths = [
        (f"events_threshold_{threshold_label(args.threshold)}", events_path, True, True),
        ("actor_mapping", args.actor_mapping, True, True),
        ("initial_period_shares", args.initial_period_shares, False, False),
        *(
            [
                (
                    "previous_population_method_period_shares",
                    args.previous_period_shares,
                    False,
                    False,
                )
            ]
            if args.previous_period_shares is not None
            else []
        ),
        ("grid_source", args.grid, True, True),
        *[
            (
                f"population_raster_{year}",
                path,
                year in {2013, 2017, 2020},
                True,
            )
            for year, path in zip(range(2011, 2021), raster_paths)
        ],
    ]
    manifest = pd.DataFrame(
        [
            {
                "input": label,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "used_by_period_model": period_used,
                "used_by_yearly_model": yearly_used,
            }
            for label, path, period_used, yearly_used in manifest_paths
        ]
    )
    manifest.to_csv(args.output_dir / "input_manifest.csv", index=False)

    metadata = {
        "interpretation": (
            "Population-weighted spatial conflict presence; not independently "
            "observed territorial control"
        ),
        "actor_universe": list(ACTORS),
        "actor_count": len(ACTORS),
        "meltt_threshold": args.threshold,
        "event_input": str(events_path),
        "operation_peace_spring_mapping": "Turkey",
        "is_in_default_denominator": True,
        "expanded_row_rule": (
            "RF-expanded rows remain independent; each actor is counted at "
            "most once per physical expanded row"
        ),
        "periods": [
            {"label": label, "start": start, "end": end, "raster_year": raster}
            for label, start, end, raster in PERIODS
        ],
        "yearly_raster_rule": (
            "Same-year raster for 2011-2020; 2020 raster for 2021-2024"
        ),
        "bandwidth_km": args.bandwidth_km,
        "cutoff_sigma": args.cutoff_sigma,
        "pseudo_activity_per_actor_cell": args.prior,
        "event_population_weighted": False,
        "cell_population_weighted": True,
        "cell_population_measurement": (
            "Sum of WorldPop people-per-pixel values whose centres fall within "
            "each 0.1-degree grid cell"
        ),
        "population_applied_once": True,
        "grid_cells": len(grid),
        "actor_event_occurrences_2011_2024": len(events),
        "expanded_rows_2011_2024": int(events["_expanded_row_id"].nunique()),
        "source_event_ids_2011_2024": int(events["id"].nunique()),
        "mapping_aliases": len(mapping),
        "mapping_umbrella_values": int(mapping_table["Umbrella"].dropna().nunique()),
        "unretained_labels": len(audit),
        "unretained_occurrences": int(audit["occurrences"].sum()),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    save_review_workbook(
        args.output_dir / "territory_presence_results.xlsx",
        args.threshold,
        period_shares_by_actor,
        period_shares,
        yearly_shares,
        comparison,
        summary,
        audit,
        manifest,
        previous_comparison,
        previous_summary,
    )

    period_check = period_shares[list(ACTORS)].sum(axis=1)
    yearly_check = yearly_shares[list(ACTORS)].sum(axis=1)
    if not np.allclose(period_check, 1.0, atol=1e-10):
        raise AssertionError("Period shares do not sum to one")
    if not np.allclose(yearly_check, 1.0, atol=1e-10):
        raise AssertionError("Yearly shares do not sum to one")
    if "Operation Peace Spring" in period_shares.columns:
        raise AssertionError("Operation Peace Spring must not be a standalone actor")

    print(period_shares.set_index("period").round(4).to_string(max_cols=12))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
