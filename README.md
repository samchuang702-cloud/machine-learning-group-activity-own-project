# Music Valence Classification Experiments

This repository contains the code used to reproduce the model evaluation scores for the music valence classification project.

The target is `valence_%`, converted into two classes:

- `High Valence`
- `Low Valence`

The evaluation metrics are:

- Accuracy
- Precision
- Recall
- F1 Score
- Area Under the Curve

## Files

| File | Purpose |
| --- | --- |
| `common_valence_utils.py` | Shared dataset loading, preprocessing, train/test split, AUC, and metric utilities |
| `01_baseline_logistic_regression.py` | Baseline Logistic Regression result |
| `02_pca_lda_logistic_regression.py` | PCA + Logistic Regression and LDA + Logistic Regression results |
| `03_logistic_regression_hyperparameter_tuning.py` | GridSearchCV tuning for Logistic Regression |
| `04_soft_voting_ensemble.py` | Soft Voting Ensemble using tuned LR, SVM, KNN, and Decision Tree |
| `spotify-2023.csv` | Dataset file |

## Install Dependencies

Run this command in the repository folder:

```bash
pip install pandas numpy scikit-learn
```

## Run Each Experiment

Baseline Logistic Regression:

```bash
python 01_baseline_logistic_regression.py
```

PCA and LDA Logistic Regression:

```bash
python 02_pca_lda_logistic_regression.py
```

Logistic Regression Hyperparameter Tuning:

```bash
python 03_logistic_regression_hyperparameter_tuning.py
```

Soft Voting Ensemble:

```bash
python 04_soft_voting_ensemble.py
```

## Expected Main Scores

| Stage | Accuracy | Precision | Recall | F1 Score | Area Under the Curve |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline Logistic Regression | 0.7696 | 0.7740 | 0.7678 | 0.7678 | 0.8308 |
| PCA (13) + Logistic Regression | 0.7539 | 0.7586 | 0.7520 | 0.7517 | 0.8202 |
| LDA (1) + Logistic Regression | 0.7696 | 0.7740 | 0.7678 | 0.7678 | 0.8341 |
| Logistic Regression Hyperparameter Tuning | 0.7644 | 0.7679 | 0.7627 | 0.7627 | 0.8305 |
| Soft Voting Ensemble | 0.7801 | 0.7884 | 0.7778 | 0.7774 | 0.8286 |

## Notes

The scripts use the same setup as the original project:

- `test_size=0.2`
- `random_state=42`
- stratified train/test split
- macro average for Precision, Recall, and F1 Score
- `High Valence` as the positive class for AUC
