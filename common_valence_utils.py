from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "spotify-2023.csv"
FEATURE_COLUMNS = [
    "bpm",
    "key",
    "mode",
    "danceability_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
]
TAIL_COLUMNS = [
    "bpm",
    "key",
    "mode",
    "danceability_%",
    "valence_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
]
NUMERIC_COLUMNS = [
    "bpm",
    "danceability_%",
    "energy_%",
    "acousticness_%",
    "instrumentalness_%",
    "liveness_%",
    "speechiness_%",
]
CATEGORICAL_COLUMNS = ["key", "mode"]
TARGET_COLUMN = "valence_%"
LABEL_COLUMN = "valence_label"
LABEL_ORDER = ["High Valence", "Low Valence"]
POSITIVE_LABEL = "High Valence"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    raw_lines = path.read_text(encoding="latin1").splitlines()
    parsed_rows = []
    for line in raw_lines[1:]:
        parts = line.split(",")
        while parts and parts[-1] == "":
            parts.pop()
        if len(parts) < len(TAIL_COLUMNS):
            continue
        parsed_rows.append(dict(zip(TAIL_COLUMNS, parts[-len(TAIL_COLUMNS) :])))

    dataset = pd.DataFrame(parsed_rows)
    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    valence_median = dataset[TARGET_COLUMN].median()
    dataset[LABEL_COLUMN] = np.where(
        dataset[TARGET_COLUMN] >= valence_median,
        LABEL_ORDER[0],
        LABEL_ORDER[1],
    )
    return dataset


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def build_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = load_dataset()
    features = dataset[FEATURE_COLUMNS]
    labels = dataset[LABEL_COLUMN]
    return train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )


def build_pipeline(model: object, extra_steps: list[tuple[str, object]] | None = None) -> Pipeline:
    steps = [("preprocessor", build_preprocessor())]
    if extra_steps:
        steps.extend(extra_steps)
    steps.append(("model", model))
    return Pipeline(steps=steps)


def get_auc(estimator: object, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    binary_y_test = (y_test == POSITIVE_LABEL).astype(int)

    if hasattr(estimator, "predict_proba"):
        positive_class_index = list(estimator.classes_).index(POSITIVE_LABEL)
        scores = estimator.predict_proba(x_test)[:, positive_class_index]
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x_test)
        classes = list(getattr(estimator, "classes_", []))
        if classes and classes[-1] != POSITIVE_LABEL:
            scores = -scores
    else:
        return np.nan

    return float(roc_auc_score(binary_y_test, scores))


def evaluate_estimator(model_name: str, estimator: object) -> dict[str, object]:
    x_train, x_test, y_train, y_test = build_split()
    return evaluate_estimator_on_split(model_name, estimator, x_train, x_test, y_train, y_test)


def evaluate_estimator_on_split(
    model_name: str,
    estimator: object,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)
    auc_score = get_auc(estimator, x_test, y_test)

    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Precision": round(precision_score(y_test, predictions, average="macro", zero_division=0), 4),
        "Recall": round(recall_score(y_test, predictions, average="macro", zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, predictions, average="macro", zero_division=0), 4),
        "Area Under the Curve": round(auc_score, 4) if not np.isnan(auc_score) else np.nan,
    }


def print_results(rows: list[dict[str, object]]) -> None:
    print(pd.DataFrame(rows).to_string(index=False))
