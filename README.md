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
[https://github.com/OshankKarn/ML-Assignment-2-Project.git]

## d. Models Used & Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well, achieving high accuracy (97.37%) and MCC. Linear decision boundaries effectively separate features after standard scaling. |
| **Decision Tree** | Prone to minor overfitting compared to linear models, resulting in lower accuracy (92.98%) and MCC, though still highly interpretable. |
| **KNN** | Strong performance (95.61% accuracy) due to distance-based neighbor clustering on standardized features, though slightly sensitive to feature scaling. |
| **Naive Bayes** | Solid probabilistic baseline with high recall, but assumes feature independence which slightly limits overall accuracy (93.86%). |
| **Support Vector Machine** | Achieved the highest AUC score (0.9934) and robust classification metrics, leveraging kernel separation effectively on scaled biomedical data. |

**Overall Winner:** **Support Vector Machine / Logistic Regression** (both achieved superior AUC and MCC scores on this feature space).
