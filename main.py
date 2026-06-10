#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Subject-independent EEG classification on the EEGMAT dataset.

Author: Mahtab Mohammadi
Year: 2026

Associated manuscript:
Subject-Independent EEG Classification During Mental Arithmetic:
A Comparison of Riemannian Geometry and Spatial Filtering Methods
"""

import os
import numpy as np
import mne

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from mne.decoding import CSP

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    balanced_accuracy_score, 
    f1_score, 
    roc_auc_score
)
from sklearn.base import BaseEstimator, TransformerMixin

# ============================================================
# Dataset location
# ============================================================

DATASET_PATH = "./dataset"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        "Dataset folder not found.\n"
        "Download EEGMAT from PhysioNet and place it in ./dataset"
    )

# ============================================================
# EEGMAT Dataset
#
# REST = 0
# TASK = 1
# ============================================================

files = os.listdir(DATASET_PATH)

subjects = sorted(
    set(
        f.split("_")[0]
        for f in files
        if f.endswith(".edf")
    )
)

print(f"Subjects found: {len(subjects)}")


X_all = []
y_all = []
groups_all = []

for sub in subjects:

    try:

        rest_f = next(
            (
                f for f in files
                if sub in f and "_1" in f and f.endswith(".edf")
            ),
            None
        )

        task_f = next(
            (
                f for f in files
                if sub in f and "_2" in f and f.endswith(".edf")
            ),
            None
        )

        if rest_f is None or task_f is None:
            continue

        for fname, label in [(rest_f, 0), (task_f, 1)]:

            raw = mne.io.read_raw_edf(
                os.path.join(DATASET_PATH, fname),
                preload=True,
                verbose=False
            )

            raw.set_eeg_reference(
                "average",
                verbose=False
            )

            raw.drop_channels(
                [
                    ch for ch in raw.ch_names
                    if "ECG" in ch or "A2-A1" in ch
                ]
            )

            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=2,
                overlap=0,
                verbose=False
            )

            X = epochs.get_data()

            X_all.append(X)
            y_all.append(np.full(len(X), label))
            groups_all.append(np.full(len(X), sub))

    except Exception as e:

        print(f"Skipping {sub}: {e}")

X_all = np.vstack(X_all)
y_all = np.hstack(y_all)
groups_all = np.hstack(groups_all)

print("X shape:", X_all.shape)
print("Class balance:", np.bincount(y_all.astype(int)))

# ============================================================
# Evaluation
# ============================================================

def evaluate(clf, X, y, groups):

    gkf = GroupKFold(n_splits=5)

    acc_scores = []
    bal_acc_scores = []
    f1_scores = []
    auc_scores = []

    for train_idx, test_idx in gkf.split(X, y, groups):

        clf.fit(X[train_idx], y[train_idx])

        y_pred = clf.predict(X[test_idx])

        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X[test_idx])[:, 1]
        else:
            y_prob = clf.decision_function(X[test_idx])

        acc_scores.append(
            np.mean(y_pred == y[test_idx])
        )

        bal_acc_scores.append(
            balanced_accuracy_score(
                y[test_idx],
                y_pred
            )
        )

        f1_scores.append(
            f1_score(
                y[test_idx],
                y_pred,
                average="macro"
            )
        )

        auc_scores.append(
            roc_auc_score(
                y[test_idx],
                y_prob
            )
        )

    return {
        "Accuracy":
            f"{np.mean(acc_scores)*100:.2f}% ± "
            f"{np.std(acc_scores)*100:.2f}%",

        "Balanced Accuracy":
            f"{np.mean(bal_acc_scores)*100:.2f}% ± "
            f"{np.std(bal_acc_scores)*100:.2f}%",

        "F1 (macro)":
            f"{np.mean(f1_scores)*100:.2f}% ± "
            f"{np.std(f1_scores)*100:.2f}%",

        "ROC-AUC":
            f"{np.mean(auc_scores):.4f} ± "
            f"{np.std(auc_scores):.4f}"
    }

# ============================================================
# FBCSP
# ============================================================

# Theta, Alpha, Beta, Gamma
FREQ_BANDS = [
    (4, 8),
    (8, 13),
    (13, 30),
    (30, 45)
]


class BandCSP(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        l_freq,
        h_freq,
        sfreq=500,
        n_components=4
    ):

        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.n_components = n_components

    def fit(self, X, y=None):

        X_filt = mne.filter.filter_data(
            X,
            self.sfreq,
            self.l_freq,
            self.h_freq,
            verbose=False
        )

        self.csp_ = CSP(
            n_components=self.n_components,
            log=True
        )

        self.csp_.fit(X_filt, y)

        return self

    def transform(self, X):

        X_filt = mne.filter.filter_data(
            X,
            self.sfreq,
            self.l_freq,
            self.h_freq,
            verbose=False
        )

        return self.csp_.transform(X_filt)

# ============================================================
# Classification Pipelines
# ============================================================

pipelines = {

    "Linear SVM + Riemannian":

        Pipeline([

            ("cov", Covariances(estimator="lwf")),

            ("ts", TangentSpace()),

            ("sc", StandardScaler()),

            ("clf", SVC(
                kernel="linear",
                C=1,
                class_weight="balanced",
                probability=True,
                random_state=42
            ))
        ]),

    "LogReg + Riemannian":

        Pipeline([

            ("cov", Covariances(estimator="lwf")),

            ("ts", TangentSpace()),

            ("sc", StandardScaler()),

            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

    "Riemannian + RBF SVM":

        Pipeline([

            ("cov", Covariances(estimator="lwf")),

            ("ts", TangentSpace()),

            ("sc", StandardScaler()),

            ("clf", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42
            ))
        ]),

    "FBCSP":

        Pipeline([

            (
                "fbcsp",

                FeatureUnion([
                    (
                        f"band_{l}_{h}",
                        BandCSP(
                            l_freq=l,
                            h_freq=h
                        )
                    )
                    for l, h in FREQ_BANDS
                ])
            ),

            ("sc", StandardScaler()),

            ("clf", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42
            ))
        ]),

    "CSP + LDA":

        Pipeline([

            (
                "csp",

                CSP(
                    n_components=8,
                    log=True
                )
            ),

            ("clf", LinearDiscriminantAnalysis())
        ])
}

# ============================================================
# Run Evaluation
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)

    for name, clf in pipelines.items():

        print(f"\n>>> {name}")

        results = evaluate(
            clf,
            X_all,
            y_all,
            groups_all
        )

        for metric, value in results.items():

            print(
                f"    {metric:22s}: {value}"
            )

    print("=" * 65)

