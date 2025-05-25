# SVM Parameter Optimization

This project focuses on tuning hyperparameters for an SVM (Support Vector Machine) model using `RandomizedSearchCV` from scikit-learn. 

---

## Dataset

Used a multi-class classification dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). Features were scaled and used directly for SVM classification.

---

## ⚙️ Methodology

- **Model**: Support Vector Classifier (`SVC`)
- **Train-Test Split**: 70-30 ratio
- **Repetition**: 10 different random seeds (`random_state = 0 to 9`)
- **Search Strategy**: `RandomizedSearchCV`
  - **Hyperparameter Space**:
    - `C`: 10 values in log space (0.01 to 100)
    - `kernel`: `'linear'`, `'rbf'`, `'poly'`, `'sigmoid'`
    - `gamma`: `'scale'`, `'auto'`
  - **Iterations**: 20 per run
  - **CV Folds**: 3
  - **Scoring Metric**: Accuracy

---

## Results

  Sample  Best Accuracy                               Best Parameters
0     S1          96.73  {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}
1     S2          91.68   {'kernel': 'rbf', 'gamma': 'scale', 'C': 1}
2     S3          96.32  {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}
3     S4          96.17  {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}
4     S5          95.92  {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}
5     S6          96.08  {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}
6     S7          95.82  {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}
7     S8          96.40  {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}
8     S9          96.52  {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}
9    S10          96.35  {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}


---

## Plot (Convergence of Best Accuracy)

The plot below shows the accuracy trend across all 10 randomized samples.

![download (1)](https://github.com/user-attachments/assets/3ee5fa83-4e02-4587-8c5a-e22e17b51f0f)


- **X-axis**: Sample iteration (S1 to S10)
- **Y-axis**: Accuracy
- **Insight**: Model consistently reaches ~97% accuracy using `rbf` kernel and appropriate `C`.


