# Project README

## Overview

This project consists of several Jupyter notebooks designed to analyze and model the risk factors for coronary heart disease (MICHD) using logistic regression. The notebooks perform different tasks related to data exploration, feature selection, model training, and evaluation. Each notebook is associated with a Python file containing necessary functions, ensuring compatibility through the analysis process.

## Jupyter Notebooks Description

### 1. **data_analysis_baseline.ipynb**

This notebook conducts an initial exploratory data analysis on the dataset and develops a baseline model and different finetuning strategies:

**1. Initial data exploration**

- Label distribution and missing values were analyzed.
- Relevant features were manually selected based on clinical significance, including chronic health indicators, lifestyle factors, and demographics.
- Features were organized in a dictionary structure detailing their types, missing value indicators, and mapping values for consistent preprocessing.

**2. Data preprocessing**
To streamline the preprocessing of the selected features:

* Features were organized in a  **dictionary structure** , detailing each feature's **variable type** (binary, categorical, or numerical) to determine the appropriate preprocessing steps.
* **Missing value indicators** were established, where specific values (e.g., 7 and 9) were designated as missing information for binary variables.
* Any **mapping values** required for certain features were identified (e.g., in some numeric variables, the value 88 indicated “none”).

**3. Baseline model development**
Model performance was evaluated using **K-fold cross-validation** with K=10:

* **Feature selection** : only features with no missing (NaN) values were retained for model training.
* **Missing value strategy** : rows with missing values (excluding NaNs) were eliminated from the dataset.

**4. Finetuning model (1)**
Model performance was evaluated using **K-fold cross-validation** with K=10:

* **Data rebalancing according to label distribution**
* **Stochastic Gradient Descent** was implemented
* **Median imputation** for continuous numeric features missing values

### 2. **feature_relevance.ipynb**

This notebook assesses the relevance of selected features for predicting MICHD:

- **Initial Evaluation**:  each feature was evaluated individually to determine its ability to predict MICHD.
- **Transformations**:  features were transformed when no significant relationship was observed. Transformations included binary conversion, consistent scaling, new feature creation, and capping extreme values.
- **Iterative Process**:  the evaluation and transformation cycle was repeated until features demonstrated a clear relationship with the target variable or were deemed irrelevant.

### 3. **feature_ablation.ipynb**

The feature ablation process involved systematically adding features to the model:

- **Sequential feature addition**: features were added one at a time, assessing their impact on the F1 score.
- **Evaluation of impact**: Each addition was evaluated, and features that improved model performance were retained while others were removed.

### 4. **hyperparameter_tuning.ipynb and grid_search.ipynb**

This notebooks explores hyperparameter tuning through grid search.

### 6. **final_model.ipynb**

The notebook develops the final model, achieved through implementing all findings in the previous notebooks:

- **Integration of selected preprocessed features**: utilizes refined features identified through prior analyses.
- **Model training**: trains the logistic regression model on the balanced dataset with optimized hyperparameters.
- **Performance evaluation**: evaluates model performance using K-fold cross-validation and F1 score metric.
- 

## Associated Python Files

Each notebook is associated with a Python file of the same name (e.g., `data_analysis_baseline.py`, `feature_relevance.py`, etc.). These files contain the functions and methods utilized in the notebooks. Functions were modified across different notebooks, resulting in separate files for each to avoid incompatibilities.

## Run Submission Script
Extract `data/dataset.zip` to a folder named `dataset` at the root of the project before running `run.py`

The `run.py` script generates a submission file based on the final model's predictions. It utilizes the trained model to make predictions on the test dataset and creates a  CSV file for submission.
