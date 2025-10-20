
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

# --- Load Artifacts ---
model = joblib.load('fraud_model.joblib')
scaler = joblib.load('scaler.joblib')
model_columns = joblib.load('model_columns.joblib')
target_encoding_maps = joblib.load('target_encoding_maps.joblib')
price_mapping = joblib.load('price_mapping.joblib')
days_mapping = joblib.load('days_mapping.joblib')
days_claim_mapping = joblib.load('days_claim_mapping.joblib')
numerical_features = joblib.load('numerical_features.joblib')

# --- Load Challenge Data ---
df = pd.read_csv('challenge_dataset.csv')
X_full = df.drop('FraudFound_P', axis=1)

# --- Feature Engineering Function ---
def process_data(df):
    X = df.copy()
    X['IsWeekendAccident'] = X['DayOfWeek'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    X['VehiclePrice_numerical'] = X['VehiclePrice'].map(price_mapping)
    X['Days_Policy_Accident_numerical'] = X['Days_Policy_Accident'].map(days_mapping)
    X['Days_Policy_Claim_numerical'] = X['Days_Policy_Claim'].map(days_claim_mapping)
    X['ClaimDelay'] = (X['Days_Policy_Claim_numerical'] - X['Days_Policy_Accident_numerical']).abs()
    X['PriceToDeductibleRatio'] = X['VehiclePrice_numerical'] / (X['Deductible'] + 1e-9)
    X = X.drop(['VehiclePrice_numerical', 'Days_Policy_Accident_numerical', 'Days_Policy_Claim_numerical'], axis=1)

    for col, mapping in target_encoding_maps.items():
        X[col + '_encoded'] = X[col].map(mapping)
        mean_value = np.mean(list(mapping.values)) # Note: This line might cause an error if mapping.values is not iterable or if mapping is empty. Assuming it's a dict-like object.
        X.loc[:, col + '_encoded'] = X[col + '_encoded'].fillna(mean_value)

    X = X.drop(list(target_encoding_maps.keys()), axis=1)

    categorical_features = X.select_dtypes(include=['object']).columns
    
    X_cat = pd.get_dummies(X[categorical_features], drop_first=True)
    
    data_to_scale = X[numerical_features]
    X_num = pd.DataFrame(scaler.transform(data_to_scale), columns=numerical_features, index=X.index)
    
    X_processed = pd.concat([X_num, X_cat], axis=1)
    
    missing_cols = set(model_columns) - set(X_processed.columns)
    for c in missing_cols:
        X_processed[c] = 0
    X_processed = X_processed[model_columns]
    
    return X_processed

# --- Tiering Function ---
def assign_tier(probability):
    if probability > 0.70:
        return 'Tier 1: URGENT'
    elif probability > 0.20:
        return 'Tier 2: REVIEW'
    else:
        return 'Tier 3: AUTO-APPROVE'

# --- Process and Predict ---
X_processed_full = process_data(X_full)
y_pred_proba_full = model.predict_proba(X_processed_full)[:, 1]

# --- Tiered Analysis ---
results_df = df.copy()
results_df['Fraud Probability'] = y_pred_proba_full
results_df['Tier'] = results_df['Fraud Probability'].apply(assign_tier)

# --- Final Verification Step ---
print("\n--- Final Tier Count Verification ---")
print("This is the exact tier summary you should see in the Streamlit app:")
print(results_df['Tier'].value_counts())
