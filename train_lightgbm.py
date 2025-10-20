import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

# Load the data
df = pd.read_csv('fraud_oracle_with_features.csv')

# Separate features and target
X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- Corrected Target Encoding ---
# Create a temporary dataframe for encoding
train_df_for_encoding = pd.concat([X_train, y_train], axis=1)

for col in ['Make', 'AccidentArea', 'PolicyType']:
    # Calculate mean on the temporary dataframe
    target_mean = train_df_for_encoding.groupby(col)['FraudFound_P'].mean()
    # Apply to training data
    X_train[col + '_encoded'] = X_train[col].map(target_mean)
    # Apply to testing data
    X_test[col + '_encoded'] = X_test[col].map(target_mean)
    # Fill NaNs in test set
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

# Align columns after one-hot encoding
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

# Make predictions
y_pred = model.predict(X_test_processed)
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

# Evaluate the model
print('--- Model Evaluation ---')
print(classification_report(y_test, y_pred))
print('--- Confusion Matrix ---')
print(confusion_matrix(y_test, y_pred))
print('\n--- AUC-ROC Score ---')
print(roc_auc_score(y_test, y_pred_proba))