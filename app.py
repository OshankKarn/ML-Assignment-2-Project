import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="WILP M.Tech AIML - ML Assignment 2", layout="wide")

st.title("S1-26_AIMLZG565- Machine Learning Assignment 2")
st.title("**Course:** Machine Learning")
st.write("**Name/ID:** Oshank Karn")
st.write("**BITS ID:** 2026ak05247")

# Sidebar for controls
st.sidebar.header("Configuration Panel")
# Download Button For Test data 
st.sidebar.subheader("Test Data")
if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button(
            label="📥 Download test_data.csv",
            data=f,
            file_name="test_data.csv",
            mime="text/csv",
            help="Click to download the test dataset required for evaluation."
        )
else:
    st.sidebar.warning("test_data.csv not found in repository.")

st.sidebar.markdown("---")
# Model Selection Dropdown
model_options = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

selected_model_name = st.sidebar.selectbox("Select Classification Model", list(model_options.keys()))

# File Uploader for Test Data
st.sidebar.subheader("Test Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

@st.cache_resource
def load_artifacts(filename):
    model_path = os.path.join("model", filename)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

scaler_path = os.path.join("model", "scaler.joblib")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# Main Panel
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success("Test dataset loaded successfully!")
    
    with st.expander("Preview Uploaded Test Data"):
        st.dataframe(test_df.head())
        
    if 'target' not in test_df.columns:
        st.error("Uploaded CSV must contain the 'target' column.")
    else:
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        # Load selected model
        model_filename = model_options[selected_model_name]
        model = load_artifacts(model_filename)
        
        if model is None:
            st.error(f"Model file {model_filename} not found in 'model/' directory.")
        else:
            # Preprocess / Predict (without retraining)
            if selected_model_name in ["Logistic Regression", "KNN", "Support Vector Machine"] and scaler is not None:
                X_test_processed = scaler.transform(X_test)
            else:
                X_test_processed = X_test
                
            y_pred = model.predict(X_test_processed)
            y_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred, dtype=float)
            
            # Compute Metrics
            acc = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.0
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            st.subheader(f"Evaluation Results for: {selected_model_name}")
            
            # Metric Display Cards
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            st.markdown("---")
            
            # Confusion Matrix & Classification Report
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)
                
            with col_b:
                st.markdown("### Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

else:
    st.info("👈 Please upload your `test_data.csv` file using the sidebar to view evaluation results and metrics.")
    st.markdown("""
    ### Quick Instructions:
    1. Select your model from the sidebar dropdown.
    2. Upload the `test_data.csv` file generated during your training pipeline.
    3. View live computed metrics, confusion matrix, and classification reports instantly!
    """)
