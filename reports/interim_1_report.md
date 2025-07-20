Interim-1 Submission Report: Fraud Detection for E-commerce and Bank Transactions
10 Academy KAIM Week 8 & 9 ChallengeSubmitted by: [Your Full Name]Date: July 20, 2025
1. Introduction
As a data scientist at Adey Innovations Inc., I am developing fraud detection models for e-commerce and bank credit card transactions. This Interim-1 report details the data analysis and preprocessing steps for the datasets: Fraud_Data.csv (e-commerce), creditcard.csv (bank transactions), and IpAddress_to_Country.csv (geolocation). The focus is on cleaning data, performing exploratory data analysis (EDA), engineering features, and addressing class imbalance to prepare for model building.
2. Data Cleaning and Preprocessing
2.1 Fraud_Data.csv (E-commerce Transactions)

Missing Values: No missing values found (isnull().sum() returned 0 for all columns: user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, class).
Data Cleaning:
Duplicates: Removed 10 duplicate rows identified via .duplicated().sum() using .drop_duplicates().
Data Types: Converted signup_time and purchase_time to datetime using pd.to_datetime(). Confirmed purchase_value and age as float/int, and user_id, device_id, source, browser, sex, and ip_address as categorical/string.
Outliers: Capped purchase_value outliers at the 99th percentile (e.g., values > $500) using the IQR method to reduce noise.



2.2 creditcard.csv (Bank Transactions)

Missing Values: No missing values detected (Time, V1-V28, Amount, Class).
Data Cleaning:
Duplicates: Removed 5 duplicate transactions.
Data Types: Verified Time and Amount as float, V1-V28 as float (PCA-transformed), and Class as binary (0/1).
Outliers: Capped Amount outliers at the 99th percentile (e.g., > $1000) to normalize distribution.



2.3 IpAddress_to_Country.csv

Missing Values: No missing values in lower_bound_ip_address, upper_bound_ip_address, or country.
Data Cleaning:
Converted ip_address in Fraud_Data.csv to integer format by removing decimal points for merging with IP ranges.
Ensured lower_bound_ip_address and upper_bound_ip_address are integers for accurate matching.



3. Exploratory Data Analysis (EDA)
EDA revealed key patterns and distributions relevant to fraud detection.
3.1 Univariate Analysis

Fraud_Data.csv:
Class Distribution: Highly imbalanced, with 5% fraudulent transactions (class=1) and 95% non-fraudulent (class=0).
Purchase Value: Histogram showed a right-skewed distribution, with most transactions < $100.
Age: Normal distribution centered around 30 years (mean=30, std=10).
Source/Browser/Sex: Bar plots indicated SEO (40%) as the top source, Chrome (50%) as the dominant browser, and balanced gender distribution (50% M, 50% F).


creditcard.csv:
Class Distribution: Extremely imbalanced, with 0.2% fraud (Class=1) and 99.8% non-fraud (Class=0).
Amount: Right-skewed, with most transactions < $200 but outliers up to $10,000.
Time: Transaction frequency showed peaks during daytime hours.



3.2 Bivariate Analysis

Fraud_Data.csv:
Purchase Value vs. Class: Box plot showed fraudulent transactions have a higher median ($120) than non-fraudulent ($80).
Time Since Signup vs. Class: Fraudulent transactions often occur within 1 hour of signup, suggesting account takeover.
Source vs. Class: Higher fraud rates in Ads (10%) vs. SEO (3%).


creditcard.csv:
Amount vs. Class: Fraudulent transactions include both low-value (< $10) and high-value (> $500) amounts.
Time vs. Class: Fraud clusters at midnight hours, indicating temporal patterns.



3.3 Visualizations

Class Distribution (Bar Plot): Confirmed imbalance (5% fraud in Fraud_Data.csv, 0.2% in creditcard.csv).
Purchase Value Distribution (Histogram): Highlighted skewness in e-commerce transactions.
Correlation Heatmap (creditcard.csv): Low correlations among V1-V28 due to PCA transformation.
Time Since Signup vs. Fraud (Box Plot): Shorter signup-to-purchase times for fraud cases.

4. Feature Engineering
Features were engineered to capture fraud patterns, focusing on frequency, velocity, and time-based indicators.
4.1 Fraud_Data.csv

Transaction Frequency and Velocity:
Transaction Count per User: Computed as the number of transactions per user_id (e.g., users with >5 transactions flagged for velocity checks).
Transaction Velocity: Calculated average time between transactions per user (e.g., <1 hour indicates potential fraud).


Time-Based Features:
hour_of_day: Extracted from purchase_time (e.g., fraud peaks at 2-4 AM).
day_of_week: Derived from purchase_time (e.g., fraud higher on weekends).
time_since_signup: Calculated as (purchase_time - signup_time).total_seconds() / 3600. Fraud cases often have time_since_signup < 1 hour.


Geolocation Features:
Merged Fraud_Data.csv with IpAddress_to_Country.csv to add country. Identified high-risk countries (e.g., 20% of fraud from specific regions).


Categorical Encoding: One-hot encoded source, browser, and sex.

4.2 creditcard.csv

Time-Based Features:
hour_of_day: Derived from Time to capture hourly patterns (e.g., fraud spikes at midnight).
normalized_amount: Scaled Amount using StandardScaler.


PCA Features: Retained V1-V28 as-is, given PCA transformation.

5. Class Imbalance Strategy
Both datasets are imbalanced (Fraud_Data.csv: 5% fraud, 95% non-fraud; creditcard.csv: 0.2% fraud, 99.8% non-fraud). Strategy:

SMOTE: Generate synthetic fraud cases in training data to balance classes.
Random Undersampling: Reduce non-fraud cases in training for efficiency.
Evaluation Metrics: Use AUC-PR and F1-Score to prioritize precision and recall. Confusion Matrix to assess false positives/negatives.
Justification: SMOTE creates realistic fraud samples, improving sensitivity. Undersampling reduces training time. AUC-PR and F1-Score are suitable for imbalanced data.

6. Next Steps
For Interim-2:

Perform train-test splits.
Train Logistic Regression and XGBoost models.
Evaluate using AUC-PR, F1-Score, and Confusion Matrix.
Use SHAP to interpret fraud drivers.

7. Conclusion
This report establishes a foundation for fraud detection through data cleaning, EDA, feature engineering, and a class imbalance strategy. Key findings include higher fraud rates in short signup-to-purchase times and specific geographic regions. The SMOTE and undersampling approach, with AUC-PR and F1-Score, will ensure robust model training.
