import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

def train_and_evaluate(X, y, dataset_name):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Handle imbalance
    X_train_res, y_train_res = handle_imbalance(X_train, y_train)
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_res, y_train_res)
    lr_pred = lr.predict_proba(X_test)[:, 1]
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_pred)
    lr_auc_pr = auc(lr_recall, lr_precision)
    lr_f1 = f1_score(y_test, lr.predict(X_test))
    
    # XGBoost
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train_res, y_train_res)
    xgb_pred = xgb.predict_proba(X_test)[:, 1]
    xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_pred)
    xgb_auc_pr = auc(xgb_recall, xgb_precision)
    xgb_f1 = f1_score(y_test, xgb.predict(X_test))
    
    # Print results
    print(f"{dataset_name} Results:")
    print(f"Logistic Regression - AUC-PR: {lr_auc_pr:.2f}, F1-Score: {lr_f1:.2f}")
    print(f"XGBoost - AUC-PR: {xgb_auc_pr:.2f}, F1-Score: {xgb_f1:.2f}")
    
    # SHAP Explainability
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(f"shap_summary_{dataset_name}.png")
    
    return xgb

if __name__ == "__main__":
    from preprocess import load_data, preprocess_fraud_data, preprocess_credit_data, handle_imbalance
    
    # Load and preprocess data
    fraud_data, ip_data, credit_data = load_data()
    fraud_data = preprocess_fraud_data(fraud_data, ip_data)
    credit_data = preprocess_credit_data(credit_data)
    
    # Fraud Data
    X_fraud = fraud_data.drop('class', axis=1)
    y_fraud = fraud_data['class']
    train_and_evaluate(X_fraud, y_fraud, "Fraud_Data")
    
    # Credit Data
    X_credit = credit_data.drop('Class', axis=1)
    y_credit = credit_data['Class']
    train_and_evaluate(X_credit, y_credit, "Credit_Data")
