"""Train and apply a random-forest model for insurgent actor prediction.

This script implements the logic from the original RF notebook in a
standalone, reproducible, and CLI-driven form. It trains a multi-label
random-forest classifier on event location and timing features, evaluates the
model by actor, and scores a larger insurgent dataset under an adjustable
probability threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit

from project_paths import CLEANED_DIR, PROJECT_ROOT

ACTORS = ["Ahrar al-Sham", "HTS: Hayat Tahrir al-Sham", "Free Syrian Army"]
FEATURE_COLUMNS = ["year", "latitude", "longitude", "month"]
DEFAULT_THRESHOLD = 0.5
SYRIAN_INSURGENT_LABEL = "syrian insurgents"
TEMPORAL_TEST_START_YEAR = 2022
N_ESTIMATORS = 100
VALIDATION_REPEATS = 10
VALIDATION_TEST_SIZE = 0.3
LOCATION_GROUP_DECIMALS = 1


def slugify_actor(actor: str) -> str:
    """Create a stable, valid column suffix for actor names."""
    return actor.lower().replace(" ", "_").replace(":", "").replace("-", "_")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a dataset and keep the real columns intact."""
    df = pd.read_csv(path, low_memory=False)
    for unnamed_col in [col for col in df.columns if col.lower().startswith("unnamed")]:
        df = df.drop(columns=[unnamed_col])
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
    return df


def prepare_training_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter to relevant actor rows and build binary actor-label columns."""
    side_cols = [col for col in frame.columns if col.startswith("Side_")]
    if not side_cols:
        raise ValueError("No Side_ columns were found in the training dataset.")

    filtered = frame[frame[side_cols].isin(ACTORS).any(axis=1)].copy()
    actor_cols = [f"is_actor_{slugify_actor(actor)}" for actor in ACTORS]

    for actor in ACTORS:
        filtered[f"is_actor_{slugify_actor(actor)}"] = (
            filtered[side_cols].isin([actor]).any(axis=1).astype(int)
        )

    return filtered[FEATURE_COLUMNS + actor_cols].copy()


def build_model_and_report(
    training_frame: pd.DataFrame,
    random_state: int = 42,
    temporal_test_start_year: int = TEMPORAL_TEST_START_YEAR,
    validation_repeats: int = VALIDATION_REPEATS,
    decision_threshold: float = DEFAULT_THRESHOLD,
) -> tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, pd.Index]:
    """Evaluate one shared multi-label RF, then fit it on all labelled rows.

    Feature-group and geographic-group validation are the primary checks for
    retrospective historical relabelling. Temporal validation is retained as a
    stress test for extrapolation to genuinely later observations.
    """
    actor_cols = [col for col in training_frame.columns if col.startswith("is_actor_")]
    if not actor_cols:
        raise ValueError("No actor target columns were generated for model training.")
    if not 0 <= decision_threshold <= 1:
        raise ValueError("decision_threshold must be between 0 and 1 inclusive")

    X = training_frame.drop(columns=actor_cols)
    y = training_frame[actor_cols]
    X = pd.get_dummies(X)

    def evaluate(
        split_name: str,
        train_index: Sequence[int],
        test_index: Sequence[int],
        repeat: int,
        validation_role: str,
    ) -> pd.DataFrame:
        X_train = X.iloc[list(train_index)]
        X_test = X.iloc[list(test_index)]
        y_train = y.iloc[list(train_index)]
        y_test = y.iloc[list(test_index)]

        # This is deliberately one shared multi-output forest. Each tree uses
        # the three-column target matrix and returns one probability per actor.
        evaluation_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=random_state + repeat,
            n_jobs=-1,
        )
        evaluation_model.fit(X_train, y_train)
        probability_outputs = evaluation_model.predict_proba(X_test)
        y_pred = np.column_stack(
            [output[:, 1] >= decision_threshold for output in probability_outputs]
        ).astype(int)

        report_frames = []
        for i, actor in enumerate(ACTORS):
            report = classification_report(
                y_test.iloc[:, i],
                y_pred[:, i],
                output_dict=True,
                zero_division=0,
            )
            report_frame = pd.DataFrame(report).T.rename_axis("label").reset_index()
            report_frame["actor"] = actor
            report_frame["split"] = split_name
            report_frame["validation_role"] = validation_role
            report_frame["repeat"] = repeat
            report_frame["decision_threshold"] = decision_threshold
            report_frame["train_rows"] = len(X_train)
            report_frame["test_rows"] = len(X_test)
            report_frames.append(report_frame)
        return pd.concat(report_frames, axis=0, ignore_index=True)

    if validation_repeats < 1:
        raise ValueError("validation_repeats must be at least 1")

    report_frames = []

    # Primary check 1: keep identical model-feature vectors together so exact
    # duplicates cannot leak across train and test.
    feature_groups = pd.util.hash_pandas_object(X, index=False)
    # Primary check 2: hold out whole 0.1-degree geographic cells.
    location_groups = pd.util.hash_pandas_object(
        X[["latitude", "longitude"]].round(LOCATION_GROUP_DECIMALS),
        index=False,
    )

    for repeat in range(validation_repeats):
        split_seed = random_state + repeat
        for split_name, groups in (
            ("feature_grouped", feature_groups),
            ("geographic_grouped", location_groups),
        ):
            group_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=VALIDATION_TEST_SIZE,
                random_state=split_seed,
            )
            train_index, test_index = next(group_splitter.split(X, y, groups=groups))
            report_frames.append(
                evaluate(
                    split_name,
                    train_index,
                    test_index,
                    repeat=repeat,
                    validation_role="primary",
                )
            )

    # Assess whether earlier observations generalize to the most recent years.
    temporal_train_index = np.flatnonzero(X["year"].to_numpy() < temporal_test_start_year)
    temporal_test_index = np.flatnonzero(X["year"].to_numpy() >= temporal_test_start_year)
    if len(temporal_train_index) == 0 or len(temporal_test_index) == 0:
        raise ValueError(
            "Temporal validation requires observations both before and from "
            f"{temporal_test_start_year} onward."
        )
    report_frames.append(
        evaluate(
            "temporal",
            temporal_train_index,
            temporal_test_index,
            repeat=0,
            validation_role="future_extrapolation_stress_test",
        )
    )

    report_df = pd.concat(report_frames, axis=0, ignore_index=True)

    positive_reports = report_df[report_df["label"] == "1"].copy()
    metric_columns = ["precision", "recall", "f1-score"]
    validation_summary = (
        positive_reports.groupby(["split", "validation_role", "actor"], sort=False)[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    validation_summary.columns = [
        "_".join(str(part) for part in column if part).replace("f1-score", "f1")
        if isinstance(column, tuple)
        else column
        for column in validation_summary.columns
    ]
    repeat_counts = (
        positive_reports.groupby(["split", "validation_role", "actor"], sort=False)
        .size()
        .rename("repeats")
        .reset_index()
    )
    validation_summary = repeat_counts.merge(
        validation_summary,
        on=["split", "validation_role", "actor"],
        how="left",
        validate="one_to_one",
    )
    validation_summary = validation_summary.fillna(0)

    # Validation models are disposable. Fit the production model on all known
    # actor-labelled rows before scoring the generic Syrian Insurgents events.
    final_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=random_state,
        n_jobs=-1,
    )
    final_model.fit(X, y)
    return final_model, report_df, validation_summary, X.columns


def get_predicted_actors(row_probs: Sequence[float], threshold: float) -> list[str]:
    """Return only the actors whose probability meets the threshold.

    Events with no actor above the threshold are dropped later, rather than
    being assigned the highest-probability actor as a fallback.
    """
    return [actor for actor, prob in zip(ACTORS, row_probs) if prob >= threshold]


def is_syrian_insurgent(value: object) -> bool:
    """Return whether a value is the generic Syrian Insurgents label."""
    return isinstance(value, str) and value.strip().casefold() == SYRIAN_INSURGENT_LABEL


def predict_insurgents(
    model: RandomForestClassifier,
    feature_columns: pd.Index,
    prediction_frame: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Threshold only Syrian Insurgents rows, while keeping all other MELTT rows intact.

    This matches the research logic: the threshold is applied to the insurgent
    label replacement step, not to all ACLED/UCDP rows in the dataset. Rows that
    do not involve Syrian Insurgents remain unchanged and are retained in the
    final output regardless of the probability cutoff.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1 inclusive")

    side_cols = [
        col
        for col in prediction_frame.columns
        if col.startswith("Side_a_") or col.startswith("Side_b_")
    ]
    if not side_cols:
        raise ValueError("No standardized Side_a_* or Side_b_* columns found in the MELTT dataset.")

    insurgent_mask = prediction_frame[side_cols].apply(
        lambda column: column.map(is_syrian_insurgent)
    ).any(axis=1)
    non_insurgent = prediction_frame[~insurgent_mask].copy()
    insurgent_rows = prediction_frame[insurgent_mask].copy()

    if insurgent_rows.empty:
        return non_insurgent.copy()

    prediction_features = insurgent_rows[FEATURE_COLUMNS].copy()
    prediction_encoded = pd.get_dummies(prediction_features)
    aligned = prediction_encoded.reindex(columns=feature_columns, fill_value=0)

    probs_list = model.predict_proba(aligned)
    prob_matrix = np.array([probs_list[i][:, 1] for i in range(len(ACTORS))]).T

    prob_data = {f"prob_{actor}": probs_list[i][:, 1] for i, actor in enumerate(ACTORS)}
    prediction_probabilities = pd.DataFrame(prob_data, index=aligned.index)
    prediction_probabilities["Predicted_actors"] = [
        get_predicted_actors(row, threshold) for row in prob_matrix
    ]
    prediction_probabilities["n_actors_predicted"] = prediction_probabilities["Predicted_actors"].apply(len)

    expanded_rows: list[pd.Series] = []
    for row_pos, row in insurgent_rows.reset_index(drop=True).iterrows():
        predicted_actors = prediction_probabilities.iloc[row_pos]["Predicted_actors"]
        if not predicted_actors:
            continue

        for actor in predicted_actors:
            new_row = row.copy()
            standardized_matches = [
                side_col for side_col in side_cols if is_syrian_insurgent(new_row[side_col])
            ]
            for side_col in standardized_matches:
                new_row[side_col] = actor

            # Keep the original lowercase UCDP actor fields consistent when
            # they contain the same generic label. Downstream monadic and
            # dyadic construction uses the standardized columns above.
            for raw_side_col in ("side_a", "side_b"):
                if raw_side_col in new_row.index and is_syrian_insurgent(new_row[raw_side_col]):
                    new_row[raw_side_col] = actor

            for probability_column in prob_data:
                new_row[probability_column] = prediction_probabilities.iloc[row_pos][probability_column]
            new_row["Predicted_actors"] = predicted_actors
            new_row["n_actors_predicted"] = len(predicted_actors)
            expanded_rows.append(new_row)

    if expanded_rows:
        replaced_insurgents = pd.DataFrame(expanded_rows)
        return pd.concat([non_insurgent, replaced_insurgents], ignore_index=True)

    return non_insurgent.copy()


def artifact_stem(threshold: float) -> str:
    """Return the canonical naming stem for the thresholded MELTT output.

    We intentionally use the full-source format `dataset_threshold_{threshold}_meltt`
    because the output is meant to be the MELTT dataset with Syrian Insurgents rows
    replaced by their specific insurgent actor labels, while preserving both ACLED
    and UCDP rows.
    """
    return f"dataset_threshold_{threshold}_meltt"


def save_outputs(
    report_df: pd.DataFrame,
    validation_summary: pd.DataFrame,
    scored_frame: pd.DataFrame,
    threshold: float,
    output_dir: Path,
) -> None:
    """Persist model evaluation and thresholded MELTT outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem(threshold)

    report_df.round(3).to_csv(
        output_dir / f"{stem}_classification_report.csv",
        index=False,
    )
    validation_summary.round(3).to_csv(
        output_dir / f"{stem}_validation_summary.csv",
        index=False,
    )
    scored_frame.to_csv(output_dir / f"{stem}.csv", index=False)


def main(threshold: float = DEFAULT_THRESHOLD) -> None:
    """Train the RF model, score insurgent rows, and report threshold summaries."""
    training_path = CLEANED_DIR / "dataset_meltt.csv"
    # IMPORTANT: use the full MELTT source as the input and apply the threshold
    # only to rows where the actor is "Syrian Insurgents". All other ACLED/UCDP
    # rows remain untouched, even when they do not exceed the probability cutoff.
    prediction_path = CLEANED_DIR / "dataset_meltt.csv"
    output_dir = PROJECT_ROOT / "01 - Data" / "02 - Source_Data-cleaned"
    stem = artifact_stem(threshold)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_df = load_dataset(training_path)
    training_data = prepare_training_data(training_df)

    model, report_df, validation_summary, feature_columns = build_model_and_report(
        training_data,
        decision_threshold=threshold,
    )
    model_path = output_dir / f"{stem}_model.pkl"
    columns_path = output_dir / f"{stem}_train_columns.pkl"
    joblib.dump(model, model_path)
    joblib.dump(list(feature_columns), columns_path)

    prediction_df = load_dataset(prediction_path)
    prediction_df["date"] = pd.to_datetime(prediction_df["date"], errors="coerce")
    prediction_df["month"] = prediction_df["date"].dt.month
    prediction_df["day"] = prediction_df["date"].dt.day

    scored_frame = predict_insurgents(model, feature_columns, prediction_df, threshold)
    save_outputs(report_df, validation_summary, scored_frame, threshold, output_dir)

    print(f"Saved model: {model_path}")
    print(f"Saved train columns: {columns_path}")
    print(f"Saved classification report: {output_dir / f'{stem}_classification_report.csv'}")
    print(f"Saved validation summary: {output_dir / f'{stem}_validation_summary.csv'}")
    print(f"Saved prediction output: {output_dir / f'{stem}.csv'}")
    print(f"Rows: {len(scored_frame)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and apply the insurgent actor random-forest model with a configurable threshold."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Probability threshold used to assign actors to each event. Default: 0.5",
    )
    args = parser.parse_args()
    main(threshold=args.threshold)
