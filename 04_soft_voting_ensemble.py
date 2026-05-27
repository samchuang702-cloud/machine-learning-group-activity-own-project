from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common_valence_utils import RANDOM_STATE, build_pipeline, evaluate_estimator, print_results


def main() -> None:
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
    row = evaluate_estimator("Soft Voting Ensemble", ensemble)
    print_results([row])


if __name__ == "__main__":
    main()
