import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib

print("Loading and processing all data for final model training...")

# --- Load Data ---
df = pd.read_csv('fraud_oracle.csv')
y = df['FraudFound_P']
X = df.drop('FraudFound_P', axis=1)

# --- Feature Engineering ---
# (Same as before)
X['IsWeekendAccident'] = X['DayOfWeek'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
price_mapping = {
    'less than 20000': 10000, '20000 to 29000': 25000, '30000 to 39000': 35000,
    '40000 to 59000': 50000, '60000 to 69000': 65000, 'more than 69000': 75000
}
X['VehiclePrice_numerical'] = X['VehiclePrice'].map(price_mapping)
days_mapping = {
    'none': 0, '1 to 7': 4, '8 to 15': 11.5, '15 to 30': 22.5, 'more than 30': 35
}
X['Days_Policy_Accident_numerical'] = X['Days_Policy_Accident'].map(days_mapping)
days_claim_mapping = {
    'none': 0, '8 to 15': 11.5, '15 to 30': 22.5, 'more than 30': 35
}
X['Days_Policy_Claim_numerical'] = X['Days_Policy_Claim'].map(days_claim_mapping)
X['ClaimDelay'] = (X['Days_Policy_Claim_numerical'] - X['Days_Policy_Accident_numerical']).abs()
X['PriceToDeductibleRatio'] = X['VehiclePrice_numerical'] / (X['Deductible'] + 1e-9)
X = X.drop(['VehiclePrice_numerical', 'Days_Policy_Accident_numerical', 'Days_Policy_Claim_numerical'], axis=1)

# --- Target Encoding ---
target_encoding_maps = {}
columns_to_encode = ['Make', 'AccidentArea', 'PolicyType']
encoding_df = pd.concat([X, y], axis=1)
for col in columns_to_encode:
    target_mean = encoding_df.groupby(col)['FraudFound_P'].mean()
    X[col + '_encoded'] = X[col].map(target_mean)
    target_encoding_maps[col] = target_mean
X = X.drop(columns_to_encode, axis=1)

# --- Preprocessing ---
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns.tolist() # Get as list

X_cat = pd.get_dummies(X[categorical_features], drop_first=True)

# Use the defined numerical_features list to ensure order
data_to_scale = X[numerical_features]
scaler = StandardScaler()
X_num = pd.DataFrame(scaler.fit_transform(data_to_scale), columns=numerical_features)

X_processed = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)

# --- Final Model Training ---
print("Training final model on all data...")
scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
final_model = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
final_model.fit(X_processed, y)

# --- Save Artifacts ---
print("Saving model artifacts...")
joblib.dump(final_model, 'fraud_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X_processed.columns.tolist(), 'model_columns.joblib')
joblib.dump(target_encoding_maps, 'target_encoding_maps.joblib')
joblib.dump(price_mapping, 'price_mapping.joblib')
joblib.dump(days_mapping, 'days_mapping.joblib')
joblib.dump(days_claim_mapping, 'days_claim_mapping.joblib')
joblib.dump(numerical_features, 'numerical_features.joblib') # <-- NEW ARTIFACT

print("All model artifacts have been saved successfully.")