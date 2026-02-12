# Machine Learning Assignment -- 2

**Course:** Machine Learning\
**Program:** M.Tech (AIML/DSE) -- BITS Pilani WILP\
**Student Name:** VISWANATHA REDDY M\
**Student ID:** 2025AA05375

------------------------------------------------------------------------

# a) Problem Statement

The objective of this assignment is to implement and compare six
classification models on a single dataset and deploy them using a
Streamlit web application.

The task is to classify whether a breast tumor is **Malignant** or
**Benign** using diagnostic measurement features.

------------------------------------------------------------------------

# b) Dataset Description \[1 Mark\]

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic)\
**Original Source:** UCI Machine Learning Repository\
**Kaggle Hosting:**
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

**Number of Instances:** 569\
**Number of Features:** 30 numerical features\
**Target Variable:** Diagnosis (Malignant / Benign)

The dataset contains features computed from digitized images of breast
mass cell nuclei.

The dataset satisfies assignment requirements: - ≥ 12 features ✔ - ≥ 500
instances ✔

------------------------------------------------------------------------

# c) Models Used and Evaluation Metrics \[6 Marks\]

The following six classification models were implemented on the same
dataset:

1.  Logistic Regression\
2.  Decision Tree Classifier\
3.  K-Nearest Neighbor (KNN)\
4.  Gaussian Naive Bayes\
5.  Random Forest (Ensemble)\
6.  XGBoost (Ensemble)

All models were trained using an 80--20 stratified train-test split.

## Evaluation Metrics Used

-   Accuracy\
-   AUC Score\
-   Precision\
-   Recall\
-   F1 Score\
-   Matthews Correlation Coefficient (MCC)

------------------------------------------------------------------------

## Model Comparison Table

  -----------------------------------------------------------------------------
  ML Model Name    Accuracy   AUC      Precision   Recall   F1 Score   MCC
  ---------------- ---------- -------- ----------- -------- ---------- --------
  Logistic         0.9649     0.9964   0.9750      0.9286   0.9512     0.9245
  Regression                                                           

  Decision Tree    0.9386     0.9425   0.9268      0.9048   0.9157     0.8676

  K-Nearest        0.9386     0.9813   0.9730      0.8571   0.9114     0.8688
  Neighbors                                                            

  Gaussian Naive   0.9298     0.9921   0.9474      0.8571   0.9000     0.8487
  Bayes                                                                

  Random Forest    0.9737     0.9934   1.0000      0.9286   0.9630     0.9442
  (Ensemble)                                                           

  XGBoost          0.9737     0.9960   1.0000      0.9286   0.9630     0.9442
  (Ensemble)                                                           
  -----------------------------------------------------------------------------

------------------------------------------------------------------------

## Best Model for Each Metric

-   **Accuracy:** Random Forest (0.9737)\
-   **AUC Score:** Logistic Regression (0.9964)\
-   **Precision:** Random Forest (1.0000)\
-   **Recall:** Logistic Regression (0.9286)\
-   **F1 Score:** Random Forest (0.9630)\
-   **MCC Score:** Random Forest (0.9442)

------------------------------------------------------------------------

# Observations on Model Performance \[3 Marks\]

  -----------------------------------------------------------------------
  ML Model Name        Observation about Model Performance
  -------------------- --------------------------------------------------
  Logistic Regression  Achieved highest AUC score (0.9964) and strong
                       recall (0.9286), indicating excellent separability
                       and balanced classification capability.

  Decision Tree        Delivered good performance but slightly lower
                       generalization compared to ensemble methods, with
                       moderate MCC (0.8676).

  K-Nearest Neighbors  High precision (0.9730) but relatively lower
                       recall (0.8571), indicating sensitivity to
                       neighbor selection and class imbalance.

  Gaussian Naive Bayes Strong AUC (0.9921) but lower recall compared to
                       Logistic Regression, influenced by independence
                       assumption.

  Random Forest        Achieved highest overall Accuracy (0.9737),
  (Ensemble)           Precision (1.0000), F1 (0.9630), and MCC (0.9442),
                       demonstrating strong ensemble stability.

  XGBoost (Ensemble)   Matched Random Forest in overall performance and
                       achieved excellent AUC (0.9960), showing strong
                       gradient boosting capability.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Streamlit Application Features \[4 Marks\]

✔ Dataset upload option (CSV -- test data only)\
✔ Model selection dropdown\
✔ Display of evaluation metrics\
✔ Confusion matrix visualization\
✔ Classification report\
✔ Prediction download functionality

------------------------------------------------------------------------

# Project Structure

    project-folder/
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── VRM_ML_Assignment_2.ipynb
    │
    ├── saved_models/
    │   ├── logistic_regression_model.pkl
    │   ├── decision_tree_model.pkl
    │   ├── knn_model.pkl
    │   ├── naive_bayes_model.pkl
    │   ├── random_forest_model.pkl
    │   ├── xgboost_model.pkl
    │   └── scaler.pkl
    │
    ├── Kaggle_Breast_Cancer_Wisconsin_data.csv
    └── test_data_for_streamlit.csv

------------------------------------------------------------------------

# How to Run

``` bash
pip install -r requirements.txt
streamlit run app.py
```

------------------------------------------------------------------------

# Declaration

This project is developed as part of Machine Learning Assignment -- 2
under BITS Pilani WILP. All implementation and experimentation were
completed independently.
