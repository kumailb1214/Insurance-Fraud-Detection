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

# --- Load Test Data ---
input_df = pd.read_csv('test_claims.csv')

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
        # mapping may be a dict or a pandas Series. Handle both.
        try:
            vals = list(mapping.values())
        except TypeError:
            vals = list(mapping.values)
        mean_value = np.mean(vals)
        X.loc[:, col + '_encoded'] = X[col + '_encoded'].fillna(mean_value)

    X = X.drop(list(target_encoding_maps.keys()), axis=1)

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    
    X_cat = pd.get_dummies(X[categorical_features], drop_first=True)

    # Only pass numerical features that the scaler was fitted on (if available)
    scaler_features = list(getattr(scaler, 'feature_names_in_', []))
    if scaler_features:
        num_feats_to_use = [f for f in numerical_features if f in scaler_features]
    else:
        num_feats_to_use = list(numerical_features)

    X_num = pd.DataFrame(scaler.transform(X[num_feats_to_use]), columns=num_feats_to_use, index=X.index)
    
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

# --- Run the Prediction ---
processed_df = process_data(input_df)
probabilities = model.predict_proba(processed_df)[:, 1]

results_df = input_df.copy()
results_df['Fraud Probability'] = probabilities
results_df['Tier'] = results_df['Fraud Probability'].apply(assign_tier)

# --- Print the Results ---
print("--- Expected Results for test_claims.csv ---")
print(results_df[['PolicyNumber', 'Make', 'FraudFound_P', 'Fraud Probability', 'Tier']])

print("\n--- Expected Tier Summary ---")
print(results_df['Tier'].value_counts())
