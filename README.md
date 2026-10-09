# Syria conflict-network pipeline

This repository contains the maintained scripts used to construct the random-forest-expanded MELTT data and the monadic and dyadic conflict-network outputs.

## Reproducibility boundary

The reproducible core starts from:

`01 - Data/02 - Source_Data-cleaned/dataset_meltt.csv`

This file is copied into a local checkout but ignored by Git because it is large and contains ACLED-derived event data. Store and share it only through an approved channel.

The repository also treats the following as frozen inputs:

- `Wiki_cleaned.csv` and `Wiki_cleaned_2324.csv`, because the maintained Wiki pairwise script starts from the cleaned files;
- `OldStanford-connections.csv`, because no maintained Python script currently recreates it;
- the Russia-amended UCDP input used by the upstream MELTT step, which is not included because the manual amendment is not fully reproducible.

The ACLED/UCDP cleaners and MELTT R script are retained for provenance, but the default production run begins with the frozen MELTT file.

## Known issue to resolve before a final public release

`02 - Data_Cleaning/acled_cleaning.py` currently maps every label containing `Communal Militia (Syria)` or `Militia (Syria)` to `Military Forces of Syria`. This is broader than the explicitly reviewed crosswalk and should be narrowed and audited before rerunning the upstream raw-data pipeline.

## Python environment

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The upstream MELTT step was last run with `meltt 0.4.3`, `readxl 1.4.3`, and `dplyr 1.1.4`; see `R-packages.txt`.

## Production run

### 1. Train and apply the random forest

```bash
python rf.py --threshold 0.5
python rf.py --threshold 0.7
python rf.py --threshold 0.8
```

### 2. Rebuild the non-MELTT pairwise inputs

```bash
python "04 - Pairwise_connections/manual_pairwise_connections.py"
python "04 - Pairwise_connections/stanford_pairwise_connections.py" --map aleppo
python "04 - Pairwise_connections/stanford_pairwise_connections.py" --map syria
python "04 - Pairwise_connections/wiki_pairwise_connections.py"
```

### 3. Build the MELTT pairwise inputs

```bash
python "04 - Pairwise_connections/meltt_pairwise_connections.py" --dataset meltt --threshold 0.5
python "04 - Pairwise_connections/meltt_pairwise_connections.py" --dataset meltt --threshold 0.7
python "04 - Pairwise_connections/meltt_pairwise_connections.py" --dataset meltt --threshold 0.8
python "04 - Pairwise_connections/meltt_pairwise_connections.py" --dataset acled --threshold 0.5
python "04 - Pairwise_connections/meltt_pairwise_connections.py" --dataset ucdp --threshold 0.5
```

### 4. Build the maintained monadic outputs

```bash
python "05 - Monadic&Dyadic/monadic_main.py" --dataset meltt --threshold 0.5
python "05 - Monadic&Dyadic/monadic_main.py" --dataset meltt --threshold 0.7
python "05 - Monadic&Dyadic/monadic_main.py" --dataset meltt --threshold 0.8
python "05 - Monadic&Dyadic/monadic_main.py" --dataset acled --threshold 0.5
python "05 - Monadic&Dyadic/monadic_main.py" --dataset ucdp --threshold 0.5
```

### 5. Build the maintained dyadic outputs

```bash
python "05 - Monadic&Dyadic/dyadic_main.py" --dataset meltt --threshold 0.5
python "05 - Monadic&Dyadic/dyadic_main.py" --dataset meltt --threshold 0.7
python "05 - Monadic&Dyadic/dyadic_main.py" --dataset meltt --threshold 0.8
python "05 - Monadic&Dyadic/dyadic_main.py" --dataset acled --threshold 0.5
python "05 - Monadic&Dyadic/dyadic_main.py" --dataset ucdp --threshold 0.5
```

The ten canonical outputs from the current production run are included in `release/`.

## Data sharing

The raw ACLED and UCDP directories are deliberately absent. UCDP data can be downloaded under its stated licence from <https://ucdp.uu.se/downloads/>. ACLED data and ACLED-derived event-level files must be handled under the applicable ACLED licence at <https://acleddata.com/eula> and should not be committed to a public repository.
