# S1-26_AIMLZG565- Machine Learning Assignment 2

## a. Problem Statement
The objective of this assignment is to implement, evaluate, and deploy five distinct machine learning classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, and Support Vector Machine) using an end-to-end MLOps workflow, culminating in an interactive Streamlit web application hosted on Streamlit Community Cloud.

## b. Dataset Description
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository / Scikit-learn datasets
- **Task:** Binary Classification (Benign vs. Malignant)
- **Instances:** 569 samples (Exceeds minimum requirement of 500)
- **Features:** 30 numerical features computed from digitized images of fine needle aspirates (Exceeds minimum requirement of 12 features)

## c. GitHub Repository Link
[Insert Your GitHub Repository URL Here]

## d. Models Used & Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9737 | 0.9912 | 0.9722 | 0.9859 | 0.9790 | 0.9431 |
| Decision Tree | 0.9298 | 0.9215 | 0.9420 | 0.9296 | 0.9358 | 0.8504 |
| KNN | 0.9561 | 0.9868 | 0.9452 | 0.9859 | 0.9652 | 0.9038 |
| Naive Bayes | 0.9386 | 0.9846 | 0.9459 | 0.9437 | 0.9448 | 0.8687 |
| Support Vector Machine | 0.9649 | 0.9934 | 0.9589 | 0.9859 | 0.9722 | 0.9242 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well, achieving high accuracy (97.37%) and MCC. Linear decision boundaries effectively separate features after standard scaling. |
| **Decision Tree** | Prone to minor overfitting compared to linear models, resulting in lower accuracy (92.98%) and MCC, though still highly interpretable. |
| **KNN** | Strong performance (95.61% accuracy) due to distance-based neighbor clustering on standardized features, though slightly sensitive to feature scaling. |
| **Naive Bayes** | Solid probabilistic baseline with high recall, but assumes feature independence which slightly limits overall accuracy (93.86%). |
| **Support Vector Machine** | Achieved the highest AUC score (0.9934) and robust classification metrics, leveraging kernel separation effectively on scaled biomedical data. |

**Overall Winner:** **Support Vector Machine / Logistic Regression** (both achieved superior AUC and MCC scores on this feature space).
