
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve
import numpy as np

# --- Re-run the training process to get the model and test data ---
df = pd.read_csv('fraud_oracle_with_features.csv')
X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
train_df_for_encoding = pd.concat([X_train, y_train], axis=1)
for col in ['Make', 'AccidentArea', 'PolicyType']:
    target_mean = train_df_for_encoding.groupby(col)['FraudFound_P'].mean()
    X_train[col + '_encoded'] = X_train[col].map(target_mean)
    X_test[col + '_encoded'] = X_test[col].map(target_mean)
    X_test[col + '_encoded'].fillna(y_train.mean(), inplace=True)
X_train = X_train.drop(['Make', 'AccidentArea', 'PolicyType'], axis=1)
X_test = X_test.drop(['Make', 'AccidentArea', 'PolicyType'], axis=1)
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

# --- Phase 3: Precision-Threshold Tuning ---

# Get precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find the threshold that gives us at least 90% recall
target_recall = 0.90
threshold_for_90_recall = thresholds[np.argmax(recall >= target_recall)]
precision_for_90_recall = precision[np.argmax(recall >= target_recall)]

print(f'To achieve a recall of at least {target_recall*100:.0f}%, you should use a probability threshold of: {threshold_for_90_recall:.4f}')
print(f'At this threshold, the precision will be {precision_for_90_recall:.4f}. This means {precision_for_90_recall*100:.1f}% of the claims you flag will actually be fraudulent.')
