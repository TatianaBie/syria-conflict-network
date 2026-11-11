#!/usr/bin/env python3
"""Run the complete period and yearly territorial-presence workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from estimate_territorial_presence import (
    ACTORS,
    IS_NAME,
    PERIODS,
    aggregate_shares,
    estimate_period,
    load_actor_events,
    load_actor_mapping,
    load_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--actor-mapping", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--population-dir", type=Path, required=True)
    parser.add_argument("--initial-period-shares", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bandwidth-km", type=float, default=20.0)
    parser.add_argument("--cutoff-sigma", type=float, default=3.0)
    parser.add_argument("--prior", type=float, default=0.05)
    return parser.parse_args()


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

    mapping, mapping_table = load_actor_mapping(args.actor_mapping)
    events, audit = load_actor_events(
        args.events,
        actor_mapping=mapping,
        deduplicate=True,
        start_year=2011,
        end_year=2024,
    )
    grid = load_grid(args.grid)
    grid.to_csv(args.output_dir / "grid_syria_0p1deg.csv", index=False)

    common = dict(
        bandwidth_km=args.bandwidth_km,
        prior_weight=args.prior,
        cutoff_sigma=args.cutoff_sigma,
        event_population_weight=True,
    )
    period_frames = [
        estimate_period(
            events,
            grid,
            args.population_dir,
            period=period,
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
        ("events", args.events, True, True),
        ("actor_mapping", args.actor_mapping, True, True),
        ("initial_period_shares", args.initial_period_shares, False, False),
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
        "event_population_weighted": True,
        "grid_population_weighted": True,
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
