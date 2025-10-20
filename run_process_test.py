import joblib
import pandas as pd
import numpy as np
from pathlib import Path

base = Path(__file__).parent
model = joblib.load(base / 'fraud_model.joblib')
scaler = joblib.load(base / 'scaler.joblib')
model_columns = joblib.load(base / 'model_columns.joblib')
target_encoding_maps = joblib.load(base / 'target_encoding_maps.joblib')
price_mapping = joblib.load(base / 'price_mapping.joblib')
days_mapping = joblib.load(base / 'days_mapping.joblib')
days_claim_mapping = joblib.load(base / 'days_claim_mapping.joblib')

input_df = pd.read_csv(base / 'test_claims.csv')
print('Loaded input_df shape', input_df.shape)

# replicate process_data
X = input_df.copy()
X['IsWeekendAccident'] = X['DayOfWeek'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
X['VehiclePrice_numerical'] = X['VehiclePrice'].map(price_mapping)
X['Days_Policy_Accident_numerical'] = X['Days_Policy_Accident'].map(days_mapping)
X['Days_Policy_Claim_numerical'] = X['Days_Policy_Claim'].map(days_claim_mapping)
X['ClaimDelay'] = (X['Days_Policy_Claim_numerical'] - X['Days_Policy_Accident_numerical']).abs()
X['PriceToDeductibleRatio'] = X['VehiclePrice_numerical'] / (X['Deductible'] + 1e-9)
X = X.drop(['VehiclePrice_numerical', 'Days_Policy_Accident_numerical', 'Days_Policy_Claim_numerical'], axis=1)

for col, mapping in target_encoding_maps.items():
    print('Processing column', col, 'mapping type', type(mapping))
    X[col + '_encoded'] = X[col].map(mapping)
    try:
        vals = list(mapping.values())
    except TypeError:
        vals = list(mapping.values)
    mean_value = np.mean(vals)
    X.loc[:, col + '_encoded'] = X[col + '_encoded'].fillna(mean_value)

X = X.drop(list(target_encoding_maps.keys()), axis=1)

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns
print('categorical_features:', categorical_features.tolist())
print('numerical_features:', numerical_features.tolist())

X_cat = pd.get_dummies(X[categorical_features], drop_first=True)
print('X_cat shape', X_cat.shape)
try:
    scaler_features = list(getattr(scaler, 'feature_names_in_', []))
except Exception:
    scaler_features = []

if scaler_features:
    num_feats_to_use = [f for f in numerical_features if f in scaler_features]
else:
    num_feats_to_use = list(numerical_features)

try:
    X_num_transformed = scaler.transform(X[num_feats_to_use])
    print('scaler.transform returned type', type(X_num_transformed))
except Exception as e:
    print('scaler.transform raised:', repr(e))
    raise
X_num = pd.DataFrame(X_num_transformed, columns=num_feats_to_use, index=X.index)
X_processed = pd.concat([X_num, X_cat], axis=1)
missing_cols = set(model_columns) - set(X_processed.columns)
print('missing_cols count', len(missing_cols))
for c in missing_cols:
    X_processed[c] = 0
X_processed = X_processed[model_columns]

probs = model.predict_proba(X_processed)[:, 1]
print('probs shape', probs.shape)

results_df = input_df.copy()
results_df['Fraud Probability'] = probs
print(results_df[['Fraud Probability']].head())
