from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

    model = build_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        extra_steps=[("lda", LinearDiscriminantAnalysis(n_components=1))],
    )
    row = evaluate_estimator_on_split(
        "LDA (1) + Logistic Regression",
        model,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    print_results([row])


if __name__ == "__main__":
    main()
