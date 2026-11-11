# Territorial presence

This folder contains the reproducible territorial-presence workflow and its
latest outputs. The model estimates population-weighted spatial conflict
presence from geolocated fighting events. It is not an independent measurement
of observed territorial control.

## Harmonisation

- The analysis uses the same 35 umbrella actors as the monadic and dyadic
  release datasets.
- `Operation Peace Spring` maps to `Turkey` and is not a standalone actor.
- The Islamic State remains in the default denominator. Separately labelled
  conditional non-IS outputs are also provided.
- RF-expanded physical rows remain independent. Within each expanded row, the
  same umbrella actor is counted at most once. This matches the monadic
  deduplication rule.
- The event input is resolved from the same canonical thresholded MELTT files
  as the monadic pipeline: `dataset_threshold_{threshold}_meltt.csv`. The
  committed results use threshold 0.5.

## Spatial specification

- Syria-interior grid: 1,835 cell centres at 0.1-degree resolution.
- Gaussian bandwidth: 20 km.
- Kernel cutoff: 60 km (three bandwidths).
- Prior: 0.05 per actor and grid cell.
- Events are weighted by WorldPop population at the event location.
- Aggregate shares are weighted by WorldPop population at grid-cell centres.
- Four-year windows use the 2013, 2017 and 2020 rasters respectively.
- Yearly estimates use the same-year raster for 2011–2020 and the 2020 raster
  for 2021–2024.

## Folder contents

- `scripts/estimate_territorial_presence.py`: reusable estimation functions and
  period-only command-line interface.
- `scripts/run_territory_presence.py`: complete period and yearly workflow.
- `results/actor_shares_by_period.csv`: primary four-year-window output.
- `results/actor_shares_yearly.csv`: yearly output for 2011–2024.
- `results/spatial_actor_allocations_by_period.csv`: period-by-cell allocations.
- `results/spatial_actor_allocations_yearly.csv`: yearly cell allocations.
- `results/*conditional_on_non_IS.csv`: explicitly conditional shares.
- `results/actor_share_comparison_with_initial.csv`: actor-level comparison with
  the initial period results.
- `results/period_comparison_with_initial.csv`: compact comparison summary.
- `results/actor_mapping_audit.csv`: labels excluded because they are not in the
  mapping or final actor universe.
- `results/input_manifest.csv`: source paths, sizes, SHA-256 checksums and usage.
- `results/run_metadata.json`: complete run specification.
- `results/territory_presence_results.xlsx`: formatted review workbook.

## External source files

The ten WorldPop GeoTIFFs total approximately 1.1 GB and each exceeds GitHub's
ordinary 100 MB file limit. They are therefore not duplicated in this Git
folder. The run used `syr_ppp_2011.tif` through `syr_ppp_2020.tif` from the
TerritoryActivity source directory. `results/input_manifest.csv` records the
exact SHA-256 checksum of every source file.

## Reproduction

From the repository root, run:

```bash
python "05 - Territory_presence/scripts/run_territory_presence.py" \
  --threshold 0.5 \
  --actor-mapping "01 - Data/Merging/Merging_names.xlsx" \
  --grid "05 - Territory_presence/results/grid_syria_0p1deg.csv" \
  --population-dir /path/to/worldpop_directory \
  --initial-period-shares /path/to/initial_actor_shares_by_period.csv \
  --output-dir "05 - Territory_presence/results"
```

The population directory must contain `syr_ppp_2011.tif` through
`syr_ppp_2020.tif`. Unless `--events` is supplied explicitly, the runner reads
`01 - Data/02 - Source_Data-cleaned/dataset_threshold_0.5_meltt.csv`. Change
`--threshold` to use another canonical monadic/dyadic threshold file.
