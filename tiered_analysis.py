import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import numpy as np

# --- Re-run the training process for our champion model ---
df = pd.read_csv('fraud_oracle_advanced_features.csv')
X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
train_df_for_encoding = pd.concat([X_train, y_train], axis=1)
columns_to_encode = ['Make', 'AccidentArea', 'PolicyType']
for col in columns_to_encode:
    target_mean = train_df_for_encoding.groupby(col)['FraudFound_P'].mean()
    X_train[col + '_encoded'] = X_train[col].map(target_mean)
    X_test[col + '_encoded'] = X_test[col].map(target_mean)
    X_test[col + '_encoded'].fillna(y_train.mean(), inplace=True)
X_train = X_train.drop(columns_to_encode, axis=1)
X_test = X_test.drop(columns_to_encode, axis=1)
categorical_features = X_train.select_dtypes(include=['object']).columns
numerical_features = X_train.select_dtypes(include=np.number).columns
X_train_cat = pd.get_dummies(X_train[categorical_features], drop_first=True)
X_test_cat = pd.get_dummies(X_test[categorical_features], drop_first=True)
X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)
scaler = StandardScaler()
X_train_num = pd.DataFrame(scaler.fit_transform(X_train[numerical_features]), columns=numerical_features)
X_test_num = pd.DataFrame(scaler.transform(X_test[numerical_features]), columns=numerical_features)
X_train_processed = pd.concat([X_train_num.reset_index(drop=True), X_train_cat.reset_index(drop=True)], axis=1)
X_test_processed = pd.concat([X_test_num.reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1)
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
model = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
model.fit(X_train_processed, y_train)
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

# --- Tiered Analysis ---
results_df = pd.DataFrame({'true_label': y_test, 'probability': y_pred_proba})

# Define thresholds
tier1_threshold = 0.70
tier2_threshold = 0.20

# Tier 1 Analysis
tier1_claims = results_df[results_df['probability'] > tier1_threshold]
total_tier1 = len(tier1_claims)
fraud_tier1 = tier1_claims['true_label'].sum()
precision_tier1 = fraud_tier1 / total_tier1 if total_tier1 > 0 else 0

# Tier 2 Analysis
tier2_claims = results_df[(results_df['probability'] <= tier1_threshold) & (results_df['probability'] > tier2_threshold)]
total_tier2 = len(tier2_claims)
fraud_tier2 = tier2_claims['true_label'].sum()
precision_tier2 = fraud_tier2 / total_tier2 if total_tier2 > 0 else 0

# Tier 3 Analysis
tier3_claims = results_df[results_df['probability'] <= tier2_threshold]
total_tier3 = len(tier3_claims)
fraud_tier3 = tier3_claims['true_label'].sum()

# Summary
total_fraud_in_test = y_test.sum()
fraud_caught_tier1 = fraud_tier1
fraud_caught_tier2 = fraud_tier2
total_fraud_caught = fraud_caught_tier1 + fraud_caught_tier2

print("--- Tiered Investigation Strategy Analysis ---")
print("\n----------------------------------------------")
print("Tier 1: URGENT (Probability > 70%)")
print(f"  - Total Claims in this Tier: {total_tier1}")
print(f"  - Actual Fraud Cases Found: {fraud_tier1}")
print(f"  - Precision (Hit Rate): {precision_tier1:.1%}")
print(f"  - Fraud Capture Rate: {fraud_caught_tier1/total_fraud_in_test:.1%} of all fraud")

print("\n----------------------------------------------")
print("Tier 2: REVIEW (Probability between 20% and 70%)")
print(f"  - Total Claims in this Tier: {total_tier2}")
print(f"  - Actual Fraud Cases Found: {fraud_tier2}")
print(f"  - Precision (Hit Rate): {precision_tier2:.1%}")
print(f"  - Fraud Capture Rate: {fraud_caught_tier2/total_fraud_in_test:.1%} of all fraud")

print("\n----------------------------------------------")
print("Tier 3: AUTO-APPROVE (Probability < 20%)")
print(f"  - Total Claims in this Tier: {total_tier3}")
print(f"  - Actual Fraud Cases Found: {fraud_tier3} (These are the missed cases)")

print("\n----------------------------------------------")
print("Overall Summary:")
print(f"By investigating Tiers 1 & 2, you would:")
print(f"  - Manually review {total_tier1 + total_tier2} claims in total.")
print(f"  - Catch {total_fraud_caught} out of {total_fraud_in_test} fraud cases ({total_fraud_caught/total_fraud_in_test:.1%} total recall).")
print("----------------------------------------------")
