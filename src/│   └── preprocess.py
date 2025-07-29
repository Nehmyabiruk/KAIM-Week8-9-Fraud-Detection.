import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

def load_data():
    fraud_data = pd.read_csv('data/Fraud_Data.csv')
    ip_data = pd.read_csv('data/IpAddress_to_Country.csv')
    credit_data = pd.read_csv('data/creditcard.csv')
    return fraud_data, ip_data, credit_data

def preprocess_fraud_data(fraud_data, ip_data):
    # Handle missing values
    fraud_data['purchase_value'].fillna(fraud_data['purchase_value'].median(), inplace=True)
    
    # Remove duplicates
    fraud_data = fraud_data.drop_duplicates()
    
    # Convert IP to integer
    fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)
    
    # Merge with IP data
    def map_ip_to_country(ip):
        for _, row in ip_data.iterrows():
            if row['lower_bound_ip_address'] <= ip <= row['upper_bound_ip_address']:
                return row['country']
        return 'Unknown'
    
    fraud_data['country'] = fraud_data['ip_address'].apply(map_ip_to_country)
    
    # Feature engineering
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    fraud_data['transaction_frequency'] = fraud_data.groupby('user_id')['purchase_time'].transform('count')
    fraud_data['velocity'] = fraud_data['purchase_value'] / (fraud_data['time_since_signup'] + 1)
    
    # Encode categorical features
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    cat_features = ['source', 'browser', 'sex', 'country']
    encoded_features = pd.DataFrame(encoder.fit_transform(fraud_data[cat_features]))
    encoded_features.columns = encoder.get_feature_names_out(cat_features)
    fraud_data = pd.concat([fraud_data.drop(cat_features, axis=1), encoded_features], axis=1)
    
    # Scale numerical features
    scaler = StandardScaler()
    num_features = ['purchase_value', 'age', 'time_since_signup', 'transaction_frequency', 'velocity']
    fraud_data[num_features] = scaler.fit_transform(fraud_data[num_features])
    
    return fraud_data

def preprocess_credit_data(credit_data):
    # Scale Amount
    scaler = StandardScaler()
    credit_data['Amount'] = scaler.fit_transform(credit_data[['Amount']])
    
    # Create interaction features
    credit_data['V1_V2'] = credit_data['V1'] * credit_data['V2']
    
    return credit_data

def handle_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

if __name__ == "__main__":
    fraud_data, ip_data, credit_data = load_data()
    fraud_data = preprocess_fraud_data(fraud_data, ip_data)
    credit_data = preprocess_credit_data(credit_data)
    print("Preprocessing complete.")
