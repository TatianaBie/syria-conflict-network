#!/usr/bin/env python3
"""Generate period and yearly territorial-presence shares."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from estimate_territorial_presence import (
    ACTORS,
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


def actor_rows(shares: pd.DataFrame, period_column: str) -> pd.DataFrame:
    """Return actors in rows and periods in columns."""
    return (
        shares.set_index(period_column)
        .reindex(columns=ACTORS)
        .T.rename_axis("umbrella_actor")
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = resolve_events_path(args)

    mapping, _ = load_actor_mapping(args.actor_mapping)
    events, _ = load_actor_events(
        events_path,
        actor_mapping=mapping,
        deduplicate=True,
        start_year=2011,
        end_year=2024,
    )
    grid = load_grid(args.grid)

    raster_years = sorted({period[3] for period in PERIODS} | set(range(2011, 2021)))
    cell_populations = {
        year: integrate_cell_population(
            args.population_dir / f"syr_ppp_{year}.tif",
            grid.lon.to_numpy(),
            grid.lat.to_numpy(),
        )
        for year in raster_years
    }
    common = {
        "bandwidth_km": args.bandwidth_km,
        "prior_weight": args.prior,
        "cutoff_sigma": args.cutoff_sigma,
    }

    period_allocations = pd.concat(
        [
            estimate_period(
                events,
                grid,
                args.population_dir,
                period=period,
                cell_population=cell_populations[period[3]],
                **common,
            )
            for period in PERIODS
        ],
        ignore_index=True,
    )
    period_shares = aggregate_shares(period_allocations)

    yearly_periods = [
        (str(year), year, year, min(year, 2020)) for year in range(2011, 2025)
    ]
    yearly_allocations = pd.concat(
        [
            estimate_period(
                events,
                grid,
                args.population_dir,
                period=period,
                cell_population=cell_populations[period[3]],
                **common,
            )
            for period in yearly_periods
        ],
        ignore_index=True,
    )
    yearly_shares = aggregate_shares(yearly_allocations).rename(
        columns={"period": "year"}
    )
    yearly_shares["year"] = yearly_shares["year"].astype(int)

    actor_rows(period_shares, "period").to_csv(
        args.output_dir / "actor_shares_by_period.csv",
        index=False,
    )
    actor_rows(yearly_shares, "year").to_csv(
        args.output_dir / "actor_shares_yearly.csv",
        index=False,
    )

    print(period_shares.set_index("period").round(4).to_string(max_cols=12))


if __name__ == "__main__":
    main()
