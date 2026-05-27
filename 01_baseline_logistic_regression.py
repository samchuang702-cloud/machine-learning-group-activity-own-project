from sklearn.linear_model import LogisticRegression

from common_valence_utils import RANDOM_STATE, build_pipeline, evaluate_estimator, print_results


def main() -> None:
    model = build_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    )
    row = evaluate_estimator("Baseline Logistic Regression", model)
    print_results([row])


if __name__ == "__main__":
    main()
