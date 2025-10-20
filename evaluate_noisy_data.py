import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

print("--- Loading Model and Noisy Data for Robustness Check ---")

# --- Load Artifacts ---
model = joblib.load('fraud_model.joblib')
scaler = joblib.load('scaler.joblib')
model_columns = joblib.load('model_columns.joblib')
target_encoding_maps = joblib.load('target_encoding_maps.joblib')
price_mapping = joblib.load('price_mapping.joblib')
days_mapping = joblib.load('days_mapping.joblib')
days_claim_mapping = joblib.load('days_claim_mapping.joblib')

# --- Load Noisy Test Data ---
df = pd.read_csv('noisy_test_data.csv')
X_full = df.drop('FraudFound_P', axis=1)
y_full = df['FraudFound_P']

# --- Feature Engineering Function (Corrected) ---
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
        mean_value = np.mean(list(mapping.values))
        X.loc[:, col + '_encoded'] = X[col + '_encoded'].fillna(mean_value)

    X = X.drop(list(target_encoding_maps.keys()), axis=1)

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    
    X_cat = pd.get_dummies(X[categorical_features], drop_first=True)
    X_num = pd.DataFrame(scaler.transform(X[numerical_features]), columns=numerical_features, index=X.index)
    
    X_processed = pd.concat([X_num, X_cat], axis=1)
    
    missing_cols = set(model_columns) - set(X_processed.columns)
    for c in missing_cols:
        X_processed[c] = 0
    X_processed = X_processed[model_columns]
    
    return X_processed

# --- Process the noisy dataset ---
print("Processing the noisy dataset...")
X_processed_full = process_data(X_full)

# --- Make Predictions ---
print("Making predictions on noisy data...")
optimal_threshold = 0.70
y_pred_proba_full = model.predict_proba(X_processed_full)[:, 1]
y_pred_full = (y_pred_proba_full >= optimal_threshold).astype(int)

# --- Evaluate Performance ---
print("\n--- Model Performance on Noisy Test Dataset ---")
print(f"(Using a threshold of {optimal_threshold:.2f})")

accuracy = accuracy_score(y_full, y_pred_full)
print(f"\nOverall Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_full, y_pred_full))

print("Confusion Matrix:")
print(confusion_matrix(y_full, y_pred_full))

auc_roc = roc_auc_score(y_full, y_pred_proba_full)
print(f"\nAUC-ROC Score: {auc_roc:.4f}")
