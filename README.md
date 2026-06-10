# EEGMAT Subject-Independent EEG Classification

This repository contains the code used in the study:

**Subject-Independent Classification of Resting-State and Mental Arithmetic EEG Using CSP, FBCSP, and Riemannian Geometry Approaches**

## Dataset

The EEGMAT dataset is publicly available from PhysioNet:

https://physionet.org/physiobank/database/eegmat/

## Methods

The repository implements:

* Linear SVM + Riemannian Geometry
* Logistic Regression + Riemannian Geometry
* Riemannian + RBF SVM
* Filter Bank Common Spatial Patterns (FBCSP)
* CSP + LDA

## Evaluation

Models are evaluated using:

* 5-fold GroupKFold cross-validation
* Accuracy
* Balanced Accuracy
* Macro F1-score
* ROC-AUC


## Author

Mahtab Mohammadi

ORCID: 0009-0005-6227-7456
