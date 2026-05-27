from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

    tuned_lr = build_pipeline(
        LogisticRegression(C=10, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
    )
    tuned_svm = build_pipeline(
        SVC(C=1, gamma="scale", kernel="rbf", probability=True, random_state=RANDOM_STATE)
    )
    tuned_knn = build_pipeline(
        KNeighborsClassifier(n_neighbors=15, weights="distance", metric="euclidean")
    )
    tuned_dt = build_pipeline(
        DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            min_samples_leaf=1,
            min_samples_split=10,
            random_state=RANDOM_STATE,
        )
    )

    ensemble = VotingClassifier(
        estimators=[
            ("lr", tuned_lr),
            ("svm", tuned_svm),
            ("knn", tuned_knn),
            ("dt", tuned_dt),
        ],
        voting="soft",
    )
    row = evaluate_estimator_on_split(
        "Soft Voting Ensemble",
        ensemble,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    print_results([row])


if __name__ == "__main__":
    main()
