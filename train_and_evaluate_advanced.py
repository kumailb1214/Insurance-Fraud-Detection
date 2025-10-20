
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import numpy as np

# Load the NEW dataset
df = pd.read_csv('fraud_oracle_advanced_features.csv')

# --- The rest of the script is the same as before ---

# Separate features and target
X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Target encoding
train_df_for_encoding = pd.concat([X_train, y_train], axis=1)
for col in ['Make', 'AccidentArea', 'PolicyType']:
    target_mean = train_df_for_encoding.groupby(col)['FraudFound_P'].mean()
    X_train[col + '_encoded'] = X_train[col].map(target_mean)
    X_test[col + '_encoded'] = X_test[col].map(target_mean)
    X_test[col + '_encoded'].fillna(y_train.mean(), inplace=True)

# Drop original categorical columns
X_train = X_train.drop(['Make', 'AccidentArea', 'PolicyType'], axis=1)
X_test = X_test.drop(['Make', 'AccidentArea', 'PolicyType'], axis=1)

# Identify categorical and numerical features
categorical_features = X_train.select_dtypes(include=['object']).columns
numerical_features = X_train.select_dtypes(include=np.number).columns

# One-hot encode categorical features
X_train_cat = pd.get_dummies(X_train[categorical_features], drop_first=True)
X_test_cat = pd.get_dummies(X_test[categorical_features], drop_first=True)
X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)

# Scale numerical features
scaler = StandardScaler()
X_train_num = pd.DataFrame(scaler.fit_transform(X_train[numerical_features]), columns=numerical_features)
X_test_num = pd.DataFrame(scaler.transform(X_test[numerical_features]), columns=numerical_features)

# Combine processed features
X_train_processed = pd.concat([X_train_num.reset_index(drop=True), X_train_cat.reset_index(drop=True)], axis=1)
X_test_processed = pd.concat([X_test_num.reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1)

# Calculate scale_pos_weight
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Train LightGBM model
model = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
model.fit(X_train_processed, y_train)

# --- Evaluation with default threshold ---
y_pred = model.predict(X_test_processed)
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

print('--- Model Evaluation with Advanced Features (Default Threshold) ---')
print(classification_report(y_test, y_pred))
print('--- Confusion Matrix ---')
print(confusion_matrix(y_test, y_pred))
print('\n--- AUC-ROC Score ---')
print(roc_auc_score(y_test, y_pred_proba))


# --- Find Best F1-Score Threshold ---
print('\n\n--- Finding the Best Balanced Threshold (Sweet Spot) ---')
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
thresholds = np.append(thresholds, 1)

best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds[best_f1_idx]
best_f1 = f1_scores[best_f1_idx]
precision_at_best_f1 = precision[best_f1_idx]
recall_at_best_f1 = recall[best_f1_idx]

print(f"\nThe best balanced performance (highest F1-score) is achieved at a threshold of: {best_f1_threshold:.4f}")
print(f"At this threshold:")
print(f"  - F1-Score: {best_f1:.4f}")
print(f"  - Recall: {recall_at_best_f1:.4f} (You will catch {recall_at_best_f1*100:.1f}% of fraud cases)")
print(f"  - Precision: {precision_at_best_f1:.4f} ( {precision_at_best_f1*100:.1f}% of flagged claims will be fraudulent)")

