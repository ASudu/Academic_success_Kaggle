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