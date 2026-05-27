from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from common_valence_utils import RANDOM_STATE, build_pipeline, evaluate_estimator, print_results


def main() -> None:
    pca_logistic = build_pipeline(
        LogisticRegression(C=1, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE),
        extra_steps=[("pca", PCA(n_components=13, random_state=RANDOM_STATE))],
    )
    lda_logistic = build_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        extra_steps=[("lda", LinearDiscriminantAnalysis(n_components=1))],
    )

    rows = [
        evaluate_estimator("PCA (13) + Logistic Regression", pca_logistic),
        evaluate_estimator("LDA (1) + Logistic Regression", lda_logistic),
    ]
    print_results(rows)


if __name__ == "__main__":
    main()
