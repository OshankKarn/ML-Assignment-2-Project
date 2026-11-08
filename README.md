# S1-26_AIMLZG565- Machine Learning Assignment 2
- **Name:** Oshank Karn
- **BITS ID:** 2026ak05247

## a. Problem Statement
The objective of this assignment is to implement, evaluate, and deploy five distinct machine learning classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, and Support Vector Machine) using an end-to-end MLOps workflow, culminating in an interactive Streamlit web application hosted on Streamlit Community Cloud.

## b. Dataset Description
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset
- **URL:** [https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data]
- **Task:** Binary Classification (Benign vs. Malignant)
- **Instances:** 569 samples (Exceeds minimum requirement of 500)
- **Features:** 30 numerical features computed from digitized images of fine needle aspirates (Exceeds minimum requirement of 12 features)
- **Test Data Download Link:**
[Download test_data.csv](https://raw.githubusercontent.com/OshankKarn/ML-Assignment-2-Project/refs/heads/main/test_data.csv)

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
| **Logistic Regression** | Showed exceptional ranking capability with an AUC of 0.9960 and strong overall accuracy (96.49%), benefiting from linear separability after scaling. |
| **Decision Tree** | Exhibited lower overall performance metrics compared to ensemble and linear models, with a reduced AUC of 0.9246 and higher variance. |
| **KNN** | Performed reliably well (95.61% accuracy) through distance-based neighborhoods, though recall was slightly lower than linear counterparts. |
| **Naive Bayes** | Achieved perfect precision (1.0000), meaning zero false positives, but suffered from lower recall (0.8333) due to its strict feature independence assumptions. |
| **Support Vector Machine** | Achieved the highest accuracy (97.37%), perfect precision (1.0000), and the highest MCC score (0.9442), demonstrating optimal margin separation on this dataset. |

**Overall Winner:** **Support Vector Machine (SVM)** — It delivered the best overall balance of high accuracy, perfect precision, and the highest Matthews Correlation Coefficient (MCC of 0.9442).

## e. **Streamlit Community cloud deployed App url:** [https://ml-assignment-2-project-m53byzdpwtvgmcxmhpt5nv.streamlit.app/]


