import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from common_valence_utils import RANDOM_STATE, build_pipeline, build_split, evaluate_estimator, print_results

warnings.filterwarnings("ignore")


def main() -> None:
    x_train, _, y_train, _ = build_split()
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

    row = evaluate_estimator("Logistic Regression Hyperparameter Tuning", grid.best_estimator_)
    row["Best Parameters"] = {
        key.replace("model__", ""): value for key, value in grid.best_params_.items()
    }
    row["Best CV F1 Macro"] = round(float(grid.best_score_), 4)
    print_results([row])


if __name__ == "__main__":
    main()
