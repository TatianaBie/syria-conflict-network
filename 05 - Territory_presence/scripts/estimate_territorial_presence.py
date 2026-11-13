#!/usr/bin/env python3
"""Estimate population-weighted spatial military presence in Syria.

Inputs
------
* Harmonized MELTT CSV with ``id``, ``year``, ``latitude``, ``longitude``
  and ``Side_a_#`` / ``Side_b_#`` actor columns.
* ``Merging_names.xlsx`` with a ``Renaming`` sheet. Source-specific actor
  names are mapped to the corresponding ``Umbrella`` value.
* CSV of Syrian grid-cell centres: ``cell_id``, ``lon``, ``lat``.
* WorldPop GeoTIFFs ``syr_ppp_2013.tif``, ``syr_ppp_2017.tif``,
  and ``syr_ppp_2020.tif`` in the same directory.

Usage
-----
python scripts/estimate_territorial_presence.py \\
    --events data/dataset_meltt_2026_rf_expanded.csv \\
    --actor-mapping data/Merging_names.xlsx \\
    --grid data/grid_syria_0p1deg.csv \\
    --population-dir data/worldpop \\
    --output-dir results/territorial_presence

Dependencies: numpy, pandas, scipy, rasterio, openpyxl. Python >= 3.10.
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
EARTH_RADIUS_KM = 6371.0
GRID_CELL_SIZE_DEGREES = 0.1
MAPPING_COLUMNS = (
    "Umbrella",
    "Stanford_Name",
    "Wiki_Name",
    "ACLED_Name",
    "Manual_Name",
)
EXPANDED_ROW_ID_COLUMN = "_expanded_row_id"

# (period label, first year, last year, available WorldPop raster year)
PERIODS = (
    ("2013-2016", 2013, 2016, 2013),
    ("2017-2020", 2017, 2020, 2017),
    ("2021-2024", 2021, 2024, 2020),
)

# Fixed harmonized actor universe, recovered from the notebook output and
# aligned to the 35-actor monadic/dyadic sample. Operation Peace Spring is an
# alias for Turkey in Merging_names.xlsx, not a separate actor.
# Actors not active in every four-year window remain in the universe.
ACTORS = (
    "Ahrar al-Sham",
    "Aknaf Bait al-Maqdis",
    "Al Qaeda",
    "Authenticity and Development Front",
    "Faylaq al-Rahman",
    "Free Syrian Army",
    "Global Coalition Against Daesh",
    "HTS: Hayat Tahrir al-Sham",
    "Harakat Nour al-Din al-Zenki",
    "Hezbollah",
    "Iran",
    "Iraq",
    "Israel",
    "JTS: Syria Liberation Front",
    "Jaish al-Muhajireen wal-Ansar",
    "Jaysh al-Fatah",
    "Jaysh al-Islam",
    "Jaysh al-Sanadeed",
    "Khalid ibn al Walid Army",
    "Kurdish Front",
    "Kurdish Islamic Front",
    "Liwa al-Tawhid",
    "Military Forces of Syria",
    "Mujahideen Army",
    "Mujahideen Shura Council",
    "Peoples Protection Units",
    "Russia",
    "Sham Legion",
    "Suqour al-Sham",
    "Syrian Democratic Forces",
    "The Islamic State",
    "The Levantine Front",
    "The Southern Front",
    "Turkey",
    "USA",
)


def normalize_actor_label(value: object) -> str | None:
    """Return a stable lookup key for a source or umbrella actor label."""
    if pd.isna(value):
        return None
    label = unicodedata.normalize("NFKC", str(value))
    label = re.sub(r"\s+", " ", label).strip()
    return label.casefold() or None


def load_actor_mapping(
    workbook_path: Path,
    *,
    sheet_name: str = "Renaming",
) -> tuple[dict[str, str], pd.DataFrame]:
    """Map every source-specific name in the workbook to its umbrella actor.

    Canonical umbrella names map to themselves. Aliases are normalized for
    Unicode, whitespace, and case. Ambiguous aliases fail loudly instead of
    being assigned according to spreadsheet row order.
    """
    if not workbook_path.exists():
        raise FileNotFoundError(f"Actor mapping workbook not found: {workbook_path}")

    table = pd.read_excel(workbook_path, sheet_name=sheet_name)
    missing = set(MAPPING_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(
            f"{workbook_path} sheet {sheet_name!r} is missing columns: {sorted(missing)}"
        )

    mapping: dict[str, str] = {}
    origins: dict[str, list[str]] = {}
    for row_index, row in table.iterrows():
        umbrella_key = normalize_actor_label(row["Umbrella"])
        if umbrella_key is None:
            continue
        umbrella = re.sub(
            r"\s+", " ", unicodedata.normalize("NFKC", str(row["Umbrella"]))
        ).strip()

        for column in MAPPING_COLUMNS:
            alias_key = normalize_actor_label(row[column])
            if alias_key is None:
                continue
            previous = mapping.get(alias_key)
            if previous is not None and previous != umbrella:
                details = ", ".join(origins.get(alias_key, []))
                raise ValueError(
                    f"Ambiguous actor alias {row[column]!r}: maps to both "
                    f"{previous!r} and {umbrella!r}. Earlier occurrences: {details}"
                )
            mapping[alias_key] = umbrella
            origins.setdefault(alias_key, []).append(f"row {row_index + 2}, {column}")

    # Keep the model's canonical actor labels valid even if a future workbook
    # omits one of them from a source-specific name column.
    for actor in ACTORS:
        key = normalize_actor_label(actor)
        if key is not None:
            mapping.setdefault(key, actor)

    LOG.info(
        "%s: loaded %s normalized aliases for %s umbrella values",
        workbook_path.name,
        f"{len(mapping):,}",
        table["Umbrella"].dropna().nunique(),
    )
    return mapping, table


def load_actor_events(
    csv_path: Path,
    *,
    actor_mapping: dict[str, str],
    deduplicate: bool = True,
    start_year: int = PERIODS[0][1],
    end_year: int = PERIODS[-1][2],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rename both sides to umbrella actors and keep two-sided events.

    By default, count a given umbrella actor at most once per physical
    RF-expanded row, including cases where several source labels in that row
    map to the same actor. Rows that share a source event ID remain independent,
    matching the monadic pipeline. Different actors in the same row are each
    retained.

    Returns the cleaned actor-event table and an audit of labels that were
    unmapped or mapped outside the fixed analysis universe.
    """
    headers = pd.read_csv(csv_path, nrows=0).columns.tolist()
    sides = [c for c in headers if re.fullmatch(r"Side_[ab]_\d+", c)]
    required = {"id", "year", "longitude", "latitude"}
    if not sides or not required.issubset(headers):
        raise ValueError(f"Missing required fields or Side_* columns in {csv_path}")

    raw = pd.read_csv(csv_path, usecols=[*sorted(required), *sides], low_memory=False)
    raw = raw.reset_index(drop=True)
    raw[EXPANDED_ROW_ID_COLUMN] = raw.index
    long = raw.melt(
        id_vars=[EXPANDED_ROW_ID_COLUMN, "id", "year", "longitude", "latitude"],
        value_vars=sides,
        var_name="side_column",
        value_name="source_actor",
    )
    long = long.dropna(subset=["id", "year", "longitude", "latitude"])
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    long = long.loc[long["year"].between(start_year, end_year)]
    long["actor_key"] = long["source_actor"].map(normalize_actor_label)
    long["actor"] = long["actor_key"].map(actor_mapping)

    nonblank = long["actor_key"].notna()
    unmapped = (
        long.loc[nonblank & long["actor"].isna(), ["source_actor"]]
        .groupby("source_actor", dropna=False)
        .size()
        .rename("occurrences")
        .reset_index()
    )
    unmapped["mapped_actor"] = pd.NA
    unmapped["status"] = "not_in_mapping"

    out_of_scope = (
        long.loc[
            nonblank & long["actor"].notna() & ~long["actor"].isin(ACTORS),
            ["source_actor", "actor"],
        ]
        .groupby(["source_actor", "actor"], dropna=False)
        .size()
        .rename("occurrences")
        .reset_index()
        .rename(columns={"actor": "mapped_actor"})
    )
    out_of_scope["status"] = "umbrella_outside_analysis_universe"

    audit = pd.concat([unmapped, out_of_scope], ignore_index=True, sort=False)
    audit = audit[["source_actor", "mapped_actor", "status", "occurrences"]]
    audit = audit.sort_values(
        ["status", "occurrences", "source_actor"], ascending=[True, False, True]
    )
    if not audit.empty:
        LOG.warning(
            "%s: %s source actor labels (%s occurrences) were not retained; "
            "see actor_mapping_audit.csv",
            csv_path.name,
            f"{len(audit):,}",
            f"{audit['occurrences'].sum():,}",
        )

    long = long.loc[long["actor"].isin(ACTORS)].copy()

    on_a = set(
        long.loc[
            long["side_column"].str.startswith("Side_a_"), EXPANDED_ROW_ID_COLUMN
        ]
    )
    on_b = set(
        long.loc[
            long["side_column"].str.startswith("Side_b_"), EXPANDED_ROW_ID_COLUMN
        ]
    )
    long = long.loc[long[EXPANDED_ROW_ID_COLUMN].isin(on_a & on_b)].copy()

    long["latitude"] = pd.to_numeric(long["latitude"], errors="coerce")
    long["longitude"] = pd.to_numeric(long["longitude"], errors="coerce")
    long = long.dropna(subset=["latitude", "longitude"])
    valid_coords = long.latitude.between(-90, 90) & long.longitude.between(-180, 180)
    long = long.loc[valid_coords]

    if deduplicate:
        before = len(long)
        long = long.drop_duplicates(subset=[EXPANDED_ROW_ID_COLUMN, "actor"])
        LOG.info("Removed %s duplicate actor-event occurrences", f"{before - len(long):,}")

    LOG.info(
        "%s: %s actor-event occurrences, %s expanded rows, %s source event IDs, %s actors",
        csv_path.name,
        f"{len(long):,}",
        f"{long[EXPANDED_ROW_ID_COLUMN].nunique():,}",
        f"{long.id.nunique():,}",
        long.actor.nunique(),
    )
    events = long[
        [EXPANDED_ROW_ID_COLUMN, "id", "year", "longitude", "latitude", "actor"]
    ].reset_index(drop=True)
    return events, audit.reset_index(drop=True)


def load_grid(grid_path: Path) -> pd.DataFrame:
    """Read the fixed Syria-interior grid, rather than approximating the border."""
    grid = pd.read_csv(grid_path)
    required = {"cell_id", "lon", "lat"}
    if not required.issubset(grid.columns):
        raise ValueError(f"Grid needs columns {sorted(required)}")
    grid = grid[["cell_id", "lon", "lat"]].drop_duplicates("cell_id")
    if grid[["lon", "lat"]].isna().any().any():
        raise ValueError("Grid has missing coordinates")
    return grid.reset_index(drop=True)


def integrate_cell_population(
    raster_path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    cell_size_degrees: float = GRID_CELL_SIZE_DEGREES,
) -> np.ndarray:
    """Sum WorldPop people-per-pixel values inside each grid cell.

    Pixel centres on a cell's western or southern edge are included and those
    on its eastern or northern edge are excluded, so adjacent cells do not
    double-count boundary pixels.
    """
    import rasterio

    if not raster_path.exists():
        raise FileNotFoundError(f"Required WorldPop raster not found: {raster_path}")
    with rasterio.open(raster_path) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            raise ValueError(f"Expected EPSG:4326 WorldPop raster: {raster_path}")
        population = ds.read(1)
        nodata = ds.nodata
        x_centres = ds.transform.c + (
            np.arange(ds.width, dtype=float) + 0.5
        ) * ds.transform.a
        y_centres = ds.transform.f + (
            np.arange(ds.height, dtype=float) + 0.5
        ) * ds.transform.e

    half_size = cell_size_degrees / 2
    y_centres_ascending = y_centres[::-1]
    totals = np.zeros(len(lon), dtype=float)
    for index, (cell_lon, cell_lat) in enumerate(zip(lon, lat)):
        left, right = cell_lon - half_size, cell_lon + half_size
        bottom, top = cell_lat - half_size, cell_lat + half_size
        column_start = np.searchsorted(x_centres, left, side="left")
        column_stop = np.searchsorted(x_centres, right, side="left")
        reverse_row_start = np.searchsorted(
            y_centres_ascending, bottom, side="left"
        )
        reverse_row_stop = np.searchsorted(
            y_centres_ascending, top, side="left"
        )
        row_start = len(y_centres) - reverse_row_stop
        row_stop = len(y_centres) - reverse_row_start
        values = population[row_start:row_stop, column_start:column_stop]
        valid = np.isfinite(values) & (values >= 0)
        if nodata is not None:
            valid &= values != nodata
        totals[index] = values[valid].sum(dtype=np.float64)
    return totals


def to_unit_sphere(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Embed WGS84 longitude/latitude on a unit sphere for radius queries."""
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    return np.column_stack((
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ))


def kernel_influence(
    grid_xyz: np.ndarray,
    event_lon: np.ndarray,
    event_lat: np.ndarray,
    event_weights: np.ndarray,
    *,
    bandwidth_km: float,
    cutoff_sigma: float,
) -> np.ndarray:
    """Sum truncated Gaussian event kernels on the grid.

    Uses a spherical KD-tree rather than materializing a huge cell-by-event
    distance matrix. Distances inside the cutoff are computed on the sphere.
    """
    from scipy.spatial import cKDTree

    result = np.zeros(len(grid_xyz), dtype=float)
    if len(event_lon) == 0:
        return result
    points = to_unit_sphere(event_lon, event_lat)
    tree = cKDTree(points)
    cutoff_km = bandwidth_km * cutoff_sigma
    chord_radius = 2 * np.sin(cutoff_km / (2 * EARTH_RADIUS_KM))
    hits_by_cell = tree.query_ball_point(grid_xyz, r=chord_radius)
    for cell_index, hits in enumerate(hits_by_cell):
        if not hits:
            continue
        # Exact great-circle (haversine-equivalent) distances in kilometres.
        cosine = np.clip(points[hits] @ grid_xyz[cell_index], -1, 1)
        distances = EARTH_RADIUS_KM * np.arccos(cosine)
        valid = distances <= cutoff_km
        result[cell_index] = np.sum(
            np.exp(-0.5 * (distances[valid] / bandwidth_km) ** 2)
            * event_weights[np.asarray(hits)[valid]]
        )
    return result


def estimate_period(
    events: pd.DataFrame,
    grid: pd.DataFrame,
    population_dir: Path,
    *,
    period: tuple[str, int, int, int],
    bandwidth_km: float,
    prior_weight: float,
    cutoff_sigma: float,
    cell_population: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return fractional spatial allocations for one four-year period."""
    label, start, stop, pop_year = period
    raster = population_dir / f"syr_ppp_{pop_year}.tif"
    current = events.loc[events.year.between(start, stop)].copy()
    cell_pop = (
        integrate_cell_population(
            raster, grid.lon.to_numpy(), grid.lat.to_numpy()
        )
        if cell_population is None
        else np.asarray(cell_population, dtype=float)
    )
    if cell_pop.sum() <= 0:
        raise ValueError(f"Population weights all zero for {label}")
    event_weights = np.ones(len(current), dtype=float)

    influence = np.full((len(grid), len(ACTORS)), prior_weight, dtype=float)
    grid_xyz = to_unit_sphere(grid.lon.to_numpy(), grid.lat.to_numpy())
    for index, actor in enumerate(ACTORS):
        selection = (current.actor == actor).to_numpy()
        if not selection.any():
            continue
        influence[:, index] += kernel_influence(
            grid_xyz,
            current.longitude.to_numpy()[selection],
            current.latitude.to_numpy()[selection],
            event_weights[selection],
            bandwidth_km=bandwidth_km,
            cutoff_sigma=cutoff_sigma,
        )

    row_total = influence.sum(axis=1)
    if np.any(row_total <= 0):
        raise ValueError("Some cells have zero actor influence; increase --prior")
    allocation = influence / row_total[:, None]
    result = grid.copy()
    result["period"] = label
    result["cell_population"] = cell_pop
    result = pd.concat([result, pd.DataFrame(allocation, columns=ACTORS)], axis=1)
    result["dominant_actor"] = np.asarray(ACTORS)[allocation.argmax(axis=1)]
    LOG.info("%s: %s actor-event occurrences; population raster %s",
             label, f"{len(current):,}", raster.name)
    return result


def aggregate_shares(grid_allocations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-cell fractions with the corresponding WorldPop weights."""
    output = []
    for period, frame in grid_allocations.groupby("period", sort=False):
        weights = frame["cell_population"].to_numpy(dtype=float)
        weights /= weights.sum()
        shares = frame.loc[:, ACTORS].to_numpy(dtype=float).T @ weights
        output.append({"period": period, **dict(zip(ACTORS, shares))})
    result = pd.DataFrame(output)
    if not np.allclose(result[list(ACTORS)].sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("Period-level actor shares must sum to 1")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, required=True, help="Harmonized MELTT events CSV")
    parser.add_argument(
        "--actor-mapping", type=Path, required=True,
        help="Merging_names.xlsx workbook containing the Renaming sheet",
    )
    parser.add_argument(
        "--actor-mapping-sheet", default="Renaming",
        help="Worksheet containing Umbrella and source-specific actor names",
    )
    parser.add_argument("--grid", type=Path, required=True, help="Syria-interior grid CSV")
    parser.add_argument("--population-dir", type=Path, required=True, help="WorldPop GeoTIFF directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results/territorial_presence"))
    parser.add_argument("--bandwidth-km", type=float, default=20.0)
    parser.add_argument("--cutoff-sigma", type=float, default=3.0)
    parser.add_argument("--prior", type=float, default=0.05, help="Pseudo-activity per actor/cell")
    duplicate_opts = parser.add_mutually_exclusive_group()
    duplicate_opts.add_argument(
        "--keep-duplicate-actor-events", dest="deduplicate_actor_event",
        action="store_false", help="Sensitivity check: reproduce the original notebook's repeated actor/event counts",
    )
    duplicate_opts.add_argument(
        "--deduplicate-actor-event", dest="deduplicate_actor_event", action="store_true",
        help="Explicitly select the default: count each actor once per RF-expanded row",
    )
    parser.set_defaults(deduplicate_actor_event=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bandwidth_km <= 0 or args.cutoff_sigma <= 0 or args.prior < 0:
        raise ValueError("Bandwidth and cutoff must be positive; prior must be nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actor_mapping, _ = load_actor_mapping(
        args.actor_mapping, sheet_name=args.actor_mapping_sheet
    )
    events, _ = load_actor_events(
        args.events,
        actor_mapping=actor_mapping,
        deduplicate=args.deduplicate_actor_event,
    )
    grid = load_grid(args.grid)
    cell_populations = {
        year: integrate_cell_population(
            args.population_dir / f"syr_ppp_{year}.tif",
            grid.lon.to_numpy(),
            grid.lat.to_numpy(),
        )
        for year in {period[3] for period in PERIODS}
    }
    frames = [
        estimate_period(
            events, grid, args.population_dir,
            period=period,
            bandwidth_km=args.bandwidth_km,
            prior_weight=args.prior,
            cutoff_sigma=args.cutoff_sigma,
            cell_population=cell_populations[period[3]],
        )
        for period in PERIODS
    ]
    result = pd.concat(frames, ignore_index=True)
    shares = aggregate_shares(result)
    actor_rows = (
        shares.set_index("period")
        .reindex(columns=ACTORS)
        .T.rename_axis("umbrella_actor")
        .reset_index()
    )
    actor_rows.to_csv(args.output_dir / "actor_shares_by_period.csv", index=False)

    LOG.info("Saved period shares to %s", args.output_dir)
    print(shares.set_index("period").round(4).to_string(max_cols=10))


if __name__ == "__main__":
    main()
