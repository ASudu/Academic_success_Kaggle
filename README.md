# Academic_success_Kaggle

## Description

This repository contains the code for the [Kaggle Playground Prediction Competetion (S4E6)](https://www.kaggle.com/competitions/playground-series-s4e6/data). In this, we deal with a classification problem of academic success - classify a student as `Graduate`, `Dropout` or `Enrolled` based on various factors including personal background (family education qualification, occupation), student's economic situation (debtor or not) and other socio-economic factors such as GDP, Inflation and Unemployment rates.

## Setup

- **Step 1:** Create a conda environment with `python 3.11.4`

```
conda create -n <env_name> python=3.11.4
```
Replace `<env_name>` with a name for the conda environment

- **Step 2:** Activate your new conda environment and execute the below command to install all the requirements

```
pip install -r requirements.txt
```

- Now you are ready to run any notebook or script file in this repository

## Dataset
- Original dataset [source](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- As mentioned in the contest page, the dataset used below is generated by DL model trained on original source

## Experiments

### Classical ML models

We pick the popular ML models like Decision Trees, Random Forest, K-Nearest Neighbors (KNN) and Naive Bayes Classifier. Since the dataset is imbalanced, we try 3 variants of each model - no pre-processing, with oversampling, with undersampling and compare the performance. The results are as follows

#### Outliers not handled

| MODEL | PRECISION (Macro avg) | RECALL (Macro avg) | F1-SCORE (Macro avg) | OVERALL ACCURACY | TRAINING TIME (in seconds) |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Decision Trees (No preprocessing) | 0.693 | 0.695 | 0.694 | 0.740 | 1.073 |
| Decision Trees (Oversampling) | 0.702 | 0.704 | 0.703 | 0.747 | 6.156 |
| Decision Trees (Undersampling) | 0.689 | 0.698 | 0.687 | 0.714 | 1.554 |
**| Random Forest (No preprocessing) | 0.800 | 0.784 | 0.791 | 0.829 | 103.134 |**
| Random Forest (Oversampling) | 0.797 | 0.786 | 0.790 | 0.826 | 316.340 |
| Random Forest (Undersampling) | 0.789 | 0.798 | 0.786 | 0.809 | 74.577 |
| Naive Bayes Classifier (No preprocessing) | 0.726 | 0.706 | 0.713 | 0.767 | 0.153 |
**| Naive Bayes Classifier (Oversampling) | 0.739 | 0.740 | 0.738 | 0.774 | 0.164 |**
| Naive Bayes Classifier (Undersampling) | 0.730 | 0.719 | 0.724 | 0.769 | 0.072 |
| KNN (No preprocessing) | 0.617 | 0.576 | 0.583 | 0.654 | 0.015 |
| KNN (Oversampling) | 0.692 | 0.691 | 0.677 | 0.696 | 0.020 |
| KNN (Undersampling) | 0.691 | 0.689 | 0.677 | 0.696 | 0.008 |

#### Outliers handled

| MODEL | PRECISION (Macro avg) | RECALL (Macro avg) | F1-SCORE (Macro avg) | OVERALL ACCURACY | TRAINING TIME (in seconds) |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Decision Trees (No preprocessing) | 0.717 | 0.721 | 0.719 | 0.762 | 1.066 |
| Decision Trees (Oversampling) | 0.722 | 0.726 | 0.724 | 0.764 | 5.182 |
| Decision Trees (Undersampling) | 0.711 | 0.727 | 0.712 | 0.736 | 1.281 |
**| Random Forest (No preprocessing) | 0.813 | 0.793 | 0.801 | 0.835 | 47.316 |**
| Random Forest (Oversampling) | 0.797 | 0.811 | 0.803 | 0.833 | 134.350 |
| Random Forest (Undersampling) | 0.807 | 0.811 | 0.799 | 0.817 | 32.307 |
| Naive Bayes Classifier (No preprocessing) | 0.719 | 0.700 | 0.708 | 0.759 | 0.093 |
**| Naive Bayes Classifier (Oversampling) | 0.730 | 0.732 | 0.730 | 0.764 | 0.101 |**
| Naive Bayes Classifier (Undersampling) | 0.722 | 0.709 | 0.714 | 0.761 | 0.038 |
| KNN (No preprocessing) | 0.722 | 0.693 | 0.704 | 0.754 | 0.014 |
| KNN (Oversampling) | 0.710 | 0.706 | 0.692 | 0.708 | 0.015 |
| KNN (Undersampling) | 0.714 | 0.708 | 0.695 | 0.712 | 0.010 |


The best models have been marked in bold - we construct an ensemble of Naive Bayes and Random Forest to see if that improves the performance 

## References
- Dataset: Realinho,Valentim, Vieira Martins,Mónica, Machado,Jorge, and Baptista,Luís. (2021). Predict Students' Dropout and Academic Success. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89. ([source](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success))
- Dataset: M.V.Martins, D. Tolledo, J. Machado, L. M.T. Baptista, V.Realinho. (2021) "Early prediction of student’s performance in higher education: a case study" Trends and Applications in Information Systems and Technologies, vol.1, in Advances in Intelligent Systems and Computing series. Springer. DOI: 10.1007/978-3-030-72657-7_16