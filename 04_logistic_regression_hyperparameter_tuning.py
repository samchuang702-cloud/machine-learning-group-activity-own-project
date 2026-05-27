import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from common_valence_utils import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    build_pipeline,
    evaluate_estimator_on_split,
    load_dataset,
    print_results,
)

warnings.filterwarnings("ignore")


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    features = dataset[FEATURE_COLUMNS]
    labels = dataset[LABEL_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline(LogisticRegression(random_state=RANDOM_STATE))
    param_grid = {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "model__solver": ["lbfgs", "liblinear"],
        "model__max_iter": [1000, 2000, 5000],
        "model__class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)

    row = evaluate_estimator_on_split(
        "Logistic Regression Hyperparameter Tuning",
        grid.best_estimator_,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    row["Best Parameters"] = {
        key.replace("model__", ""): value for key, value in grid.best_params_.items()
    }
    row["Best CV F1 Macro"] = round(float(grid.best_score_), 4)
    print_results([row])


if __name__ == "__main__":
    main()
