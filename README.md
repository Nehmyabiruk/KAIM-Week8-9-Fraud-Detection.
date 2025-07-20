KAIM Week 8 & 9: Fraud Detection for E-commerce and Bank Transactions
Project Overview
This repository supports the 10 Academy KAIM Week 8 & 9 Challenge to improve fraud detection for e-commerce and bank transactions using machine learning. The project includes data analysis, preprocessing, feature engineering, model building, and explainability with SHAP.
Interim-1 Submission (July 20, 2025)
Focus

Task 1: Data Analysis and Preprocessing
Data cleaning (missing values, duplicates, data types).
Exploratory Data Analysis (EDA) with visualizations.
Feature engineering (transaction frequency, velocity, time-based features, geolocation).
Class imbalance strategy (SMOTE, undersampling, AUC-PR, F1-Score).



Repository Structure

notebooks/: Jupyter notebooks for data cleaning, EDA, and feature engineering.
reports/: Interim-1 report (PDF).
requirements.txt: Python dependencies.
data/: Datasets (not uploaded due to 10 Academy restrictions).

Setup Instructions

Clone the repository: git clone https://github.com/[Nehmyabiruk]/KAIM-Week8-9-Fraud-Detection.git
Install dependencies: pip install -r requirements.txt
Run notebooks in notebooks/ using Jupyter.

Key Insights

Class Imbalance: Fraud_Data.csv has 5% fraud and 95% non-fraud; creditcard.csv has 0.2% fraud and 99.8% non-fraud.
Fraudulent transactions often occur within 1 hour of signup (time_since_signup).
Geographic mismatches and high transaction velocity are strong fraud indicators.

Next Steps

Train Logistic Regression and XGBoost models.
Evaluate with AUC-PR, F1-Score, and Confusion Matrix.
Apply SHAP for model explainability.

Contact
[Your Name] at [Your Email].
