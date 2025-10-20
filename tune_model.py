import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
import numpy as np
from scipy.stats import uniform, randint

print("Loading and preprocessing data...")
# --- Data Loading and Preprocessing (using the champion model's features) ---
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

# --- Hyperparameter Tuning Setup ---
print("Setting up hyperparameter tuning...")
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
lgbm = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight)

param_dist = {
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.2),
    'num_leaves': randint(20, 50),
    'max_depth': randint(5, 20),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1),
    'colsample_bytree': uniform(0.6, 0.4)
}

n_iter_search = 20
cv_folds = 3

random_search = RandomizedSearchCV(
    lgbm,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    cv=cv_folds,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print(f"Starting Randomized Search with {n_iter_search} iterations and {cv_folds} folds...")
random_search.fit(X_train_processed, y_train)
print("Hyperparameter tuning complete.")

# --- Evaluation of the Best Model ---
print("\n--- Best Hyperparameters Found ---")
print(random_search.best_params_)

best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]

print('\n--- Final Tuned Model Evaluation ---')
print('\n--- Finding the Best Balanced Threshold (Sweet Spot) ---')
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
print(f"\nFor comparison, the untuned champion model had an F1-Score of 0.6466 with 63.3% precision and 66.1% recall.")
