import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, matthews_corrcoef
)
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🏥",
    layout="wide"
)

# Header with Student Information and Logo
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <p style='margin: 0; font-size: 14px;'><strong>Student Name:</strong> VISWANATHA REDDY M</p>
        <p style='margin: 0; font-size: 14px;'><strong>Student ID:</strong> 2025AA05375</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    try:
        st.image("Data/BITS_WILP.png", width=200)
    except:
        st.warning("Logo image not found")

# Title and description
st.title("🏥 Breast Cancer Classification App")
st.markdown("""
This application predicts whether a breast cancer tumor is **Malignant** or **Benign** 
based on 30 diagnostic features from the Wisconsin Breast Cancer dataset using Machine Learning models.

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Total Instances:** 569 | **Features:** 30 | **Classes:** 2 (Benign, Malignant)
""")

# Sidebar for model selection and file upload
st.sidebar.header("Configuration")

# Model selection dropdown
model_options = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "K-Nearest Neighbors": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(model_options.keys())
)

# File upload
st.sidebar.header("Data Upload")

# Download option for test data
st.sidebar.markdown("### 📥 Download Test Data")
st.sidebar.markdown("""
**Test Dataset Details:**
- 114 samples (20% of total dataset)
- 30 features + diagnosis column
- Stratified split (maintains class distribution)
- Same data used in notebook evaluation
""")

# Try to provide download button using local file first, then GitHub
test_data_path = "Data/test_data_for_streamlit.csv"
if os.path.exists(test_data_path):
    with open(test_data_path, 'rb') as f:
        st.sidebar.download_button(
            label="⬇️ Download Test Data CSV",
            data=f,
            file_name="test_data_for_streamlit.csv",
            mime="text/csv",
            help="Download the test dataset to upload and get predictions"
        )
else:
    try:
        import requests
        raw_url = "https://raw.githubusercontent.com/2025AA05375-WILP-BITS/ML_Assignment_2_Project_Breast_Cancer/main/Data/test_data_for_streamlit.csv"
        response = requests.get(raw_url, timeout=5)
        if response.status_code == 200:
            st.sidebar.download_button(
                label="⬇️ Download Test Data CSV",
                data=response.content,
                file_name="test_data_for_streamlit.csv",
                mime="text/csv",
                help="Download the test dataset to upload and get predictions"
            )
        else:
            st.sidebar.info("Download unavailable. Please use the GitHub link above.")
    except:
        st.sidebar.info("Download unavailable. Please use the GitHub link above.")

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data (CSV)",
    type=['csv'],
    help="Upload a CSV file with test data for prediction"
)

# Load selected model and scaler
@st.cache_resource
def load_model(model_name):
    """Load the selected model and scaler"""
    try:
        model_path = f"model/{model_options[model_name]}"
        scaler_path = "model/scaler.pkl"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load the selected model
model, scaler = load_model(selected_model_name)

if model is None:
    st.error("Failed to load the model. Please check if the model files exist in the 'model' directory.")
    st.stop()

st.sidebar.success(f"✅ {selected_model_name} loaded successfully!")

# Main content
if uploaded_file is not None:
    # Read the uploaded file
    try:
        df = pd.read_csv(uploaded_file)
        
        st.header("📊 Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Check if required columns exist
        if 'diagnosis' in df.columns:
            # Separate features and target
            X_test = df.drop(['diagnosis'], axis=1)
            
            # Remove ID column if present
            if 'id' in X_test.columns:
                X_test = X_test.drop(['id'], axis=1)
            
            # Remove any unnamed columns
            X_test = X_test.loc[:, ~X_test.columns.str.contains('^Unnamed')]
            
            y_test = df['diagnosis']
            
            # Encode target variable
            y_test_encoded = y_test.map({'M': 1, 'B': 0})
            
            # Determine which data to use based on model type
            # Tree-based models (Decision Tree, Random Forest, XGBoost) use unscaled data
            # Other models (Logistic Regression, KNN, Naive Bayes) use scaled data
            tree_based_models = ["Decision Tree", "Random Forest", "XGBoost"]
            
            if selected_model_name in tree_based_models:
                # Use unscaled data for tree-based models
                X_test_final = X_test
            else:
                # Scale the features for distance-based and linear models
                X_test_final = scaler.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test_final)
            y_pred_proba = model.predict_proba(X_test_final)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Display predictions
            st.header("🔮 Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Summary")
                pred_counts = pd.Series(y_pred).value_counts()
                pred_df = pd.DataFrame({
                    'Diagnosis': ['Benign (0)', 'Malignant (1)'],
                    'Count': [pred_counts.get(0, 0), pred_counts.get(1, 0)]
                })
                st.dataframe(pred_df, use_container_width=True)
            
            with col2:
                st.subheader("Prediction Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                labels = ['Benign', 'Malignant']
                colors = ['#2ecc71', '#e74c3c']
                ax.pie(
                    [pred_counts.get(0, 0), pred_counts.get(1, 0)],
                    labels=labels,
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90
                )
                ax.set_title('Prediction Distribution')
                st.pyplot(fig)
                plt.close()
            
            # Evaluation Metrics
            st.header("📈 Evaluation Metrics")
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            precision = precision_score(y_test_encoded, y_pred, zero_division=0)
            recall = recall_score(y_test_encoded, y_pred, zero_division=0)
            f1 = f1_score(y_test_encoded, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test_encoded, y_pred)
            
            # Display metrics in columns
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("Precision", f"{precision:.4f}")
            with col3:
                st.metric("Recall", f"{recall:.4f}")
            with col4:
                st.metric("F1-Score", f"{f1:.4f}")
            with col5:
                st.metric("MCC Score", f"{mcc:.4f}")
            
            # ROC-AUC Score (if probability predictions available)
            if y_pred_proba is not None:
                try:
                    auc_score = roc_auc_score(y_test_encoded, y_pred_proba)
                    st.metric("ROC-AUC Score", f"{auc_score:.4f}")
                except:
                    pass
            
            # Confusion Matrix and Classification Report
            st.header("📊 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test_encoded, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'],
                    ax=ax,
                    cbar_kws={'label': 'Count'}
                )
                ax.set_xlabel('Predicted Label', fontsize=12)
                ax.set_ylabel('True Label', fontsize=12)
                ax.set_title(f'Confusion Matrix - {selected_model_name}', fontsize=14, fontweight='bold')
                st.pyplot(fig)
                plt.close()
                
                # Confusion matrix breakdown
                st.markdown("**Confusion Matrix Breakdown:**")
                st.write(f"- True Negatives (TN): {cm[0, 0]}")
                st.write(f"- False Positives (FP): {cm[0, 1]}")
                st.write(f"- False Negatives (FN): {cm[1, 0]}")
                st.write(f"- True Positives (TP): {cm[1, 1]}")
            
            with col2:
                st.subheader("Classification Report")
                
                # Get classification report as dictionary
                report = classification_report(
                    y_test_encoded,
                    y_pred,
                    target_names=['Benign', 'Malignant'],
                    output_dict=True,
                    zero_division=0
                )
                
                # Convert to DataFrame for better display
                report_df = pd.DataFrame(report).transpose()
                report_df = report_df.round(4)
                
                st.dataframe(report_df, use_container_width=True)
            
            # Download predictions
            st.header("💾 Download Predictions")
            
            result_df = df.copy()
            result_df['Predicted_Diagnosis'] = ['Malignant' if p == 1 else 'Benign' for p in y_pred]
            if y_pred_proba is not None:
                result_df['Prediction_Probability'] = y_pred_proba
            
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f"predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
        else:
            st.error("The uploaded file must contain a 'diagnosis' column for evaluation.")
            st.info("Please upload a CSV file with the proper format including the 'diagnosis' column.")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure the uploaded file is a valid CSV with the correct format.")

else:
    # Show instructions when no file is uploaded
    st.info("👈 **Please upload a CSV file from the sidebar to begin prediction and evaluation.**")
    st.markdown("""
    💡 **Quick Start:** Download the test data file from the sidebar and upload it to see the app in action!
    """)
    
    st.header("📋 Instructions")
    st.markdown("""
    ### How to Use This App:
    1. **Download Test Data** using the button in the sidebar (optional but recommended)
    2. **Select a Model** from the dropdown in the sidebar
    3. **Upload Test Data** (CSV file) using the file uploader
    4. The app will automatically:
       - Display the uploaded data preview
       - Make predictions using the selected model
       - Show comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score, MCC, ROC-AUC)
       - Display confusion matrix and classification report
       - Provide downloadable predictions
    
    ### Expected CSV Format:
    The CSV file must contain:
    - **Required:** A `diagnosis` column with values 'M' (Malignant) or 'B' (Benign)
    - **Required:** All 30 feature columns (listed below)
    - **Optional:** An `id` column (will be automatically ignored)
    
    ### Required Features (30 total):
    
    **Mean Measurements (10 features):**
    - radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean
    - compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean
    
    **Standard Error Measurements (10 features):**
    - radius_se, texture_se, perimeter_se, area_se, smoothness_se
    - compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se
    
    **Worst/Largest Measurements (10 features):**
    - radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst
    - compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst
    
    ### Note:
    - Download the provided test data file to ensure correct format and see expected results
    - The test data contains 114 samples with ground truth labels for evaluation
    - Models are trained on 80% of the dataset and tested on 20% (stratified split)
    """)
    
    st.header("🤖 Available Models")
    st.markdown("""
    This app includes 6 different machine learning models trained on the Breast Cancer Wisconsin dataset:
    
    | Model | Type | Description |
    |-------|------|-------------|
    | **Logistic Regression** | Linear | Probabilistic linear classifier with L2 regularization |
    | **K-Nearest Neighbors** | Instance-based | Classifies based on k=5 nearest neighbors |
    | **Naive Bayes** | Probabilistic | Gaussian Naive Bayes classifier |
    | **Decision Tree** | Tree-based | Non-linear tree-based classifier |
    | **Random Forest** | Ensemble | Ensemble of 100 decision trees |
    | **XGBoost** | Gradient Boosting | Advanced gradient boosting classifier |
    
    **Note:** 
    - Distance-based models (Logistic Regression, KNN, Naive Bayes) use scaled features
    - Tree-based models (Decision Tree, Random Forest, XGBoost) use unscaled features
    - All models were trained with random_state=42 for reproducibility
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 10px;'>
    <p><strong>Breast Cancer Classification App</strong> | Machine Learning Assignment 2</p>
    <p>Built with Streamlit | BITS Pilani WILP 2025 | Student: VISWANATHA REDDY M (2025AA05375)</p>
    <p style='font-size: 12px;'>Dataset: Wisconsin Breast Cancer (Diagnostic) | 569 instances, 30 features, 2 classes</p>
</div>
""", unsafe_allow_html=True)
