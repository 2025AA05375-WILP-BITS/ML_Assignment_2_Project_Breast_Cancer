# Machine Learning Assignment -- 2

**Course:** Machine Learning\
**Program:** M.Tech in AIML -- BITS Pilani WILP\
**Student Name:** VISWANATHA REDDY M\
**Student ID:** 2025AA05375

------------------------------------------------------------------------

# a) Problem Statement

The objective of this assignment is to implement and compare six
classification models performance for a given dataset and deploy using 
Streamlit web application.

Ojective is ot clasiffy whether a breast tumor is **Malignant** or
**Benign** using clinical diagnostic measurement features.

------------------------------------------------------------------------

# b) Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic)\
**Original Source:** UCI Machine Learning Repository\
**Kaggle Hosting:**
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

**Number of Instances:** 569\
**Number of Features:** 30 numerical features\
**Target Variable:** Diagnosis (Malignant / Benign)

The dataset contains features computed from digitized images of breast
mass cell nuclei.
------------------------------------------------------------------------

# c) Models Used and Evaluation Metrics

The following six classification models are implemented on the same
dataset:

1.  Logistic Regression\
2.  Decision Tree Classifier\
3.  K-Nearest Neighbor (KNN)\
4.  Gaussian Naive Bayes\
5.  Random Forest (Ensemble)\
6.  XGBoost (Ensemble)

All models were trained using an 80/20 train-test split with stratify.

## Evaluation Metrics Used

-   Accuracy\
-   AUC Score\
-   Precision\
-   Recall\
-   F1 Score\
-   Matthews Correlation Coefficient (MCC)

------------------------------------------------------------------------

## Model Comparison Table

| ML Model Name        | Accuracy | AUC Score | Precision | Recall   | F1 Score | MCC Score |
|----------------------|----------|-----------|-----------|----------|----------|-----------|
| Logistic Regression  | 0.9649   | 0.9960    | 0.9750    | 0.9286   | 0.9512   | 0.9245    |
| Decision Tree        | 0.9386   | 0.9425    | 0.9268    | 0.9048   | 0.9157   | 0.8676    |
| K-Nearest Neighbors  | 0.9561   | 0.9823    | 0.9744    | 0.9048   | 0.9383   | 0.9058    |
| Gaussian Naive Bayes | 0.9211   | 0.9891    | 0.9231    | 0.8571   | 0.8889   | 0.8292    |
| Random Forest        | 0.9737   | 0.9929    | 1.0000    | 0.9286   | 0.9630   | 0.9442    |
| XGBoost              | 0.9737   | 0.9921    | 1.0000    | 0.9286   | 0.9630   | 0.9442    |

------------------------------------------------------------------------

## Best Model for Each Metric

-   **Accuracy:** Random Forest & XGBoost (0.9737)\
-   **AUC Score:** Logistic Regression (0.9960)\
-   **Precision:** Random Forest & XGBoost (1.0000)\
-   **Recall:** Logistic Regression (0.9286)\
-   **F1 Score:** Random Forest & XGBoost (0.9630)\
-   **MCC Score:** Random Forest & XGBoost (0.9442)

------------------------------------------------------------------------

# Observations on Model Performance

  -----------------------------------------------------------------------
  ML Model Name        Observation about Model Performance
  -------------------- --------------------------------------------------
  Logistic Regression  Achieved highest AUC score (0.9964) and strong
                       recall (0.9286), indicating excellent separability
                       and balanced classification capability.

  Decision Tree        Delivered good performance but slightly lower
                       generalization compared to ensemble methods, with
                       moderate MCC (0.8676).

  K-Nearest Neighbors  Excellent precision (0.9744) and strong AUC
                       (0.9823) with balanced performance (F1: 0.9383),
                       demonstrating good generalization with k=5.

  Gaussian Naive Bayes Strong AUC (0.9891) but moderate recall (0.8571)
                       compared to Logistic Regression, influenced by
                       feature independence assumption.

  Random Forest        Achieved highest overall Accuracy (0.9737),
  (Ensemble)           Precision (1.0000), F1 (0.9630), and MCC (0.9442),
                       demonstrating strong ensemble stability.

  XGBoost (Ensemble)   Matched Random Forest in Accuracy, Precision, F1,
                       and MCC with excellent AUC (0.9921), demonstrating
                       powerful gradient boosting capability.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Streamlit Application Features

✔ **Model Selection:** Dropdown to choose from 6 trained models\
✔ **Test Data Download:** Download the exact test dataset (114 samples) used in evaluation\
✔ **Data Upload:** CSV file upload with validation\
✔ **Comprehensive Metrics:** Accuracy, AUC, Precision, Recall, F1-Score, MCC\
✔ **Visualizations:** 
  - Prediction distribution pie chart
  - Confusion matrix heatmap
  - Classification report table\
✔ **Results Export:** Download predictions with probabilities as CSV\
✔ **Student Information:** Header with student name and ID

------------------------------------------------------------------------

# Project Structure

    ML_Project_Breast_Cancer/
    │
    ├── app.py                              # Streamlit web application
    ├── requirements.txt                    # Python dependencies
    ├── README.md                           # Project documentation
    │
    ├── Data/
    │   ├── Kaggle_Breast_Cancer_Wisconsin_data.csv  # Original dataset (569 samples)
    │   └── test_data_for_streamlit.csv              # Test data (114 samples)
    │
    └── model/
        ├── 2025AA05375_VISWANATHA_REDDY_M_ML_Assignment_2.ipynb  # Main notebook
        ├── logistic_regression_model.pkl    # Trained Logistic Regression
        ├── decision_tree_model.pkl          # Trained Decision Tree
        ├── knn_model.pkl                    # Trained K-Nearest Neighbors
        ├── naive_bayes_model.pkl            # Trained Naive Bayes
        ├── random_forest_model.pkl          # Trained Random Forest
        ├── xgboost_model.pkl                # Trained XGBoost
        └── scaler.pkl                       # StandardScaler for feature scaling

-----------------------------------------------------------------------
