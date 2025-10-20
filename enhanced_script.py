import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('fraud_oracle_advanced_features.csv')

print("="*80)
print("ADVANCED FRAUD DETECTION - ENSEMBLE MODEL WITH FEATURE ENGINEERING")
print("="*80)

# ============================================================================
# STEP 1: AGGRESSIVE FEATURE ENGINEERING
# ============================================================================
print("\n[1/8] Creating advanced features...")

# Convert categorical to numeric properly
def convert_age_of_vehicle(val):
    if pd.isna(val) or val == 'new':
        return 0
    elif 'more than' in str(val):
        return 8
    else:
        return int(str(val).split()[0])

df['AgeOfVehicle_numeric'] = df['AgeOfVehicle'].apply(convert_age_of_vehicle)

# Convert vehicle price to numeric
def convert_vehicle_price(val):
    if 'less than' in str(val):
        return 15000
    elif '20000 to 29000' in str(val):
        return 25000
    elif '30000 to 39000' in str(val):
        return 35000
    elif '40000 to 59000' in str(val):
        return 50000
    elif '60000 to 69000' in str(val):
        return 65000
    else:  # more than 69000
        return 80000

df['VehiclePrice_numeric'] = df['VehiclePrice'].apply(convert_vehicle_price)

# Behavioral patterns
df['ClaimFrequency'] = df['PastNumberOfClaims'].replace('none', 0).astype(str)
df['ClaimFrequency'] = pd.to_numeric(df['ClaimFrequency'].str.extract('(\d+)', expand=False), errors='coerce').fillna(0)
df['ClaimFrequency'] = df['ClaimFrequency'] / (df['Age'] - 16 + 1)

df['IsNewDriver'] = ((df['Age'] - df['AgeOfVehicle_numeric']) <= 18).astype(int)
df['IsYoungDriver'] = (df['Age'] < 25).astype(int)
df['IsOldDriver'] = (df['Age'] > 60).astype(int)

# Days encoding
df['Days_Policy_Accident_numeric'] = df['Days_Policy_Accident'].replace({
    'none': 0, '1 to 7': 4, '8 to 15': 11, '15 to 30': 22, 'more than 30': 45
})
df['Days_Policy_Claim_numeric'] = df['Days_Policy_Claim'].replace({
    'none': 0, '1 to 7': 4, '8 to 15': 11, '15 to 30': 22, 'more than 30': 45
})

df['IsNewPolicy'] = (df['Days_Policy_Accident_numeric'] < 30).astype(int)
df['IsVeryNewPolicy'] = (df['Days_Policy_Accident_numeric'] < 7).astype(int)

# Suspicious timing patterns
df['QuickClaim'] = (df['ClaimDelay'] == 0).astype(int)
df['VeryQuickClaim'] = (df['ClaimDelay'] < 3).astype(int)
df['DelayedClaim'] = (df['ClaimDelay'] > 30).astype(int)

# Weekend patterns
weekend_days = ['Saturday', 'Sunday']
df['IsWeekendClaim'] = df['DayOfWeekClaimed'].isin(weekend_days).astype(int)
df['WeekendClaimWeekdayAccident'] = ((df['IsWeekendAccident'] == 0) & (df['IsWeekendClaim'] == 1)).astype(int)
df['WeekdayClaimWeekendAccident'] = ((df['IsWeekendAccident'] == 1) & (df['IsWeekendClaim'] == 0)).astype(int)

# High-risk combinations
df['HighValueLowDeductible'] = ((df['VehiclePrice_numeric'] > 60000) & (df['Deductible'] < 500)).astype(int)
df['YoungDriverSportCar'] = ((df['Age'] < 25) & (df['VehicleCategory'] == 'Sport')).astype(int)
df['YoungDriverHighValue'] = ((df['Age'] < 25) & (df['VehiclePrice_numeric'] > 50000)).astype(int)
df['OldCarHighClaim'] = ((df['AgeOfVehicle_numeric'] > 6) & (df['VehiclePrice_numeric'] > 40000)).astype(int)

# Address/policy changes - HIGH FRAUD INDICATOR
df['RecentAddressChange'] = (df['AddressChange_Claim'] == '1 year').astype(int)
df['VeryRecentAddressChange'] = (df['AddressChange_Claim'] == 'under 6 months').astype(int)
df['HasSupplements'] = (df['NumberOfSuppliments'] != 'none').astype(int)
df['ManySupplements'] = df['NumberOfSuppliments'].replace({
    'none': 0, '1 to 2': 1.5, '3 to 5': 4, 'more than 5': 7
})

# Statistical interactions
df['AgeToVehicleAge'] = df['Age'] / (df['AgeOfVehicle_numeric'] + 1)
df['DeductibleToRating'] = df['Deductible'] / (df['DriverRating'] + 1)
df['PricePerAge'] = df['VehiclePrice_numeric'] / (df['AgeOfVehicle_numeric'] + 1)
df['ClaimDelayRatio'] = df['ClaimDelay'] / (df['Days_Policy_Accident_numeric'] + 1)

# Policy holder age numeric
age_mapping = {
    '16 to 17': 16.5, '18 to 20': 19, '21 to 25': 23, '26 to 30': 28,
    '31 to 35': 33, '36 to 40': 38, '41 to 50': 45.5, '51 to 65': 58, 'over 65': 70
}
df['PolicyHolderAgeNumeric'] = df['AgeOfPolicyHolder'].map(age_mapping)
df['AgeDifference'] = abs(df['Age'] - df['PolicyHolderAgeNumeric'])
df['IsPolicyHolderYoung'] = (df['PolicyHolderAgeNumeric'] < 25).astype(int)

# Past claims numeric
df['PastNumberOfClaims_numeric'] = df['PastNumberOfClaims'].replace({
    'none': 0, '1': 1, '2 to 4': 3, 'more than 4': 6
})
df['HasPreviousClaims'] = (df['PastNumberOfClaims_numeric'] > 0).astype(int)
df['ManyPreviousClaims'] = (df['PastNumberOfClaims_numeric'] > 2).astype(int)

# Police report + witness - CRITICAL FRAUD INDICATORS
df['PoliceAndWitness'] = ((df['PoliceReportFiled'] == 'Yes') & (df['WitnessPresent'] == 'Yes')).astype(int)
df['NoPoliceNoWitness'] = ((df['PoliceReportFiled'] == 'No') & (df['WitnessPresent'] == 'No')).astype(int)
df['PoliceButNoWitness'] = ((df['PoliceReportFiled'] == 'Yes') & (df['WitnessPresent'] == 'No')).astype(int)

# Number of cars
df['ManyCars'] = df['NumberOfCars'].replace({
    '1 vehicle': 1, '2 vehicles': 2, '3 to 4': 3.5, '5 to 8': 6.5, 'more than 8': 10
})
df['SingleCar'] = (df['NumberOfCars'] == '1 vehicle').astype(int)

# Fault indicator
df['IsPolicyHolderAtFault'] = (df['Fault'] == 'Policy Holder').astype(int)
df['IsThirdPartyFault'] = (df['Fault'] == 'Third Party').astype(int)

# Agent type
df['IsExternalAgent'] = (df['AgentType'] == 'External').astype(int)

# Multi-factor risk scores
df['RiskScore1'] = (df['IsNewPolicy'] + df['NoPoliceNoWitness'] + 
                     df['RecentAddressChange'] + df['QuickClaim'])
df['RiskScore2'] = (df['YoungDriverSportCar'] + df['HighValueLowDeductible'] + 
                     df['HasPreviousClaims'] + df['IsExternalAgent'])
df['RiskScore3'] = (df['WeekendClaimWeekdayAccident'] + df['ManySupplements'] + 
                     df['IsVeryNewPolicy'] + df['VeryRecentAddressChange'])

print(f"   ✓ Created {df.shape[1] - len(pd.read_csv('fraud_oracle_advanced_features.csv').columns)} new features")

# ============================================================================
# STEP 2: PREPARE DATA
# ============================================================================
print("\n[2/8] Preparing features and target...")

X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   ✓ Training: {len(X_train)}, Test: {len(X_test)}")
print(f"   ✓ Fraud rate: {y_train.mean()*100:.1f}%")

# ============================================================================
# STEP 3: ADVANCED ENCODING
# ============================================================================
print("\n[3/8] Applying advanced encoding...")

train_df_for_encoding = pd.concat([X_train, y_train], axis=1)
high_card_features = ['Make', 'AccidentArea', 'PolicyType']

for col in high_card_features:
    # Target encoding with smoothing
    global_mean = y_train.mean()
    counts = train_df_for_encoding.groupby(col).size()
    target_sum = train_df_for_encoding.groupby(col)['FraudFound_P'].sum()
    
    # Smoothed target encoding (m-estimate)
    m = 10
    target_mean = (target_sum + m * global_mean) / (counts + m)
    
    X_train[col + '_encoded'] = X_train[col].map(target_mean)
    X_test[col + '_encoded'] = X_test[col].map(target_mean).fillna(global_mean)
    
    # Frequency encoding
    freq_encoding = X_train[col].value_counts() / len(X_train)
    X_train[col + '_freq'] = X_train[col].map(freq_encoding)
    X_test[col + '_freq'] = X_test[col].map(freq_encoding).fillna(0)
    
    # Count encoding
    count_encoding = X_train[col].value_counts()
    X_train[col + '_count'] = X_train[col].map(count_encoding)
    X_test[col + '_count'] = X_test[col].map(count_encoding).fillna(0)

X_train = X_train.drop(high_card_features, axis=1)
X_test = X_test.drop(high_card_features, axis=1)

print(f"   ✓ Applied 3 encoding methods per categorical feature")

# ============================================================================
# STEP 4: PREPROCESSING
# ============================================================================
print("\n[4/8] Preprocessing features...")

categorical_features = X_train.select_dtypes(include=['object']).columns
numerical_features = X_train.select_dtypes(include=np.number).columns

X_train_cat = pd.get_dummies(X_train[categorical_features], drop_first=True)
X_test_cat = pd.get_dummies(X_test[categorical_features], drop_first=True)
X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)

# Use RobustScaler (less sensitive to outliers)
scaler = RobustScaler()
X_train_num = pd.DataFrame(
    scaler.fit_transform(X_train[numerical_features]), 
    columns=numerical_features,
    index=X_train.index
)
X_test_num = pd.DataFrame(
    scaler.transform(X_test[numerical_features]), 
    columns=numerical_features,
    index=X_test.index
)

X_train_processed = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_processed = pd.concat([X_test_num, X_test_cat], axis=1)

print(f"   ✓ Total features: {X_train_processed.shape[1]}")

# ============================================================================
# STEP 5: ADVANCED RESAMPLING - SMOTETomek
# ============================================================================
print("\n[5/8] Applying SMOTETomek (SMOTE + Tomek Links)...")

# SMOTETomek combines oversampling (SMOTE) with undersampling (Tomek links)
# This removes noisy samples from both classes
smote_tomek = SMOTETomek(random_state=42, sampling_strategy=0.8)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_processed, y_train)

print(f"   ✓ After SMOTETomek - Fraud: {y_train_balanced.sum()}, Non-fraud: {len(y_train_balanced) - y_train_balanced.sum()}")

# ============================================================================
# STEP 6: BUILD STACKING ENSEMBLE
# ============================================================================
print("\n[6/8] Building stacking ensemble model...")

# Define base models with tuned parameters
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=50,
    max_depth=8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=1.0,
    reg_lambda=1.0,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
    random_state=42,
    eval_metric='logloss'
)

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=3.0,
    subsample=0.8,
    class_weights=[1, 3],
    random_state=42,
    verbose=0
)

# Stacking ensemble with logistic regression meta-learner
stacking_model = StackingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('xgb', xgb_model),
        ('cat', cat_model)
    ],
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
    cv=3,
    stack_method='predict_proba',
    n_jobs=-1
)

print("   ✓ Training stacking ensemble (LightGBM + XGBoost + CatBoost)...")
stacking_model.fit(X_train_balanced, y_train_balanced)
print("   ✓ Ensemble training complete!")

# ============================================================================
# STEP 7: HYPERPARAMETER TUNING FOR BEST SINGLE MODEL (LightGBM)
# ============================================================================
print("\n[7/8] Fine-tuning LightGBM for comparison...")

param_space = {
    'num_leaves': Integer(30, 80),
    'max_depth': Integer(5, 10),
    'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
    'n_estimators': Integer(300, 700),
    'min_child_samples': Integer(15, 40),
    'subsample': Real(0.7, 0.95),
    'colsample_bytree': Real(0.7, 0.95),
    'reg_alpha': Real(0.5, 3.0),
    'reg_lambda': Real(0.5, 3.0),
}

base_model = lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

opt = BayesSearchCV(
    base_model, param_space, n_iter=25, cv=cv, scoring='f1',
    n_jobs=-1, random_state=42, verbose=0
)

opt.fit(X_train_balanced, y_train_balanced)
best_lgb = opt.best_estimator_

print(f"   ✓ Best LightGBM F1 (CV): {opt.best_score_:.4f}")

# ============================================================================
# STEP 8: EVALUATE BOTH MODELS
# ============================================================================
print("\n[8/8] Evaluating models...")
print("="*80)

# Evaluate Stacking Ensemble
y_pred_proba_stack = stacking_model.predict_proba(X_test_processed)[:, 1]
precision_s, recall_s, thresholds_s = precision_recall_curve(y_test, y_pred_proba_stack)
f1_scores_s = 2 * (precision_s * recall_s) / (precision_s + recall_s + 1e-9)
thresholds_s = np.append(thresholds_s, 1)
best_f1_idx_s = np.argmax(f1_scores_s)
best_threshold_s = thresholds_s[best_f1_idx_s]
y_pred_stack = (y_pred_proba_stack >= best_threshold_s).astype(int)

# Evaluate Tuned LightGBM
y_pred_proba_lgb = best_lgb.predict_proba(X_test_processed)[:, 1]
precision_l, recall_l, thresholds_l = precision_recall_curve(y_test, y_pred_proba_lgb)
f1_scores_l = 2 * (precision_l * recall_l) / (precision_l + recall_l + 1e-9)
thresholds_l = np.append(thresholds_l, 1)
best_f1_idx_l = np.argmax(f1_scores_l)
best_threshold_l = thresholds_l[best_f1_idx_l]
y_pred_lgb = (y_pred_proba_lgb >= best_threshold_l).astype(int)

print("\n🏆 MODEL COMPARISON")
print("="*80)
print("\n📊 STACKING ENSEMBLE (LightGBM + XGBoost + CatBoost):")
print(f"   Threshold: {best_threshold_s:.4f}")
print(f"   F1-Score:  {f1_scores_s[best_f1_idx_s]:.4f}")
print(f"   Recall:    {recall_s[best_f1_idx_s]:.4f} ({recall_s[best_f1_idx_s]*100:.1f}% fraud caught)")
print(f"   Precision: {precision_s[best_f1_idx_s]:.4f} ({precision_s[best_f1_idx_s]*100:.1f}% accuracy on flags)")
print(f"   AUC-ROC:   {roc_auc_score(y_test, y_pred_proba_stack):.4f}")

print("\n📊 TUNED LIGHTGBM:")
print(f"   Threshold: {best_threshold_l:.4f}")
print(f"   F1-Score:  {f1_scores_l[best_f1_idx_l]:.4f}")
print(f"   Recall:    {recall_l[best_f1_idx_l]:.4f} ({recall_l[best_f1_idx_l]*100:.1f}% fraud caught)")
print(f"   Precision: {precision_l[best_f1_idx_l]:.4f} ({precision_l[best_f1_idx_l]*100:.1f}% accuracy on flags)")
print(f"   AUC-ROC:   {roc_auc_score(y_test, y_pred_proba_lgb):.4f}")

# Choose best model
if f1_scores_s[best_f1_idx_s] > f1_scores_l[best_f1_idx_l]:
    best_model = stacking_model
    y_pred_final = y_pred_stack
    model_name = "Stacking Ensemble"
else:
    best_model = best_lgb
    y_pred_final = y_pred_lgb
    model_name = "Tuned LightGBM"

print(f"\n🎯 BEST MODEL: {model_name}")
print("="*80)

print("\n📈 CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred_final)
print(f"   True Negatives:  {cm[0][0]:>6}")
print(f"   False Positives: {cm[0][1]:>6}")
print(f"   False Negatives: {cm[1][0]:>6}")
print(f"   True Positives:  {cm[1][1]:>6}")

print("\n📋 CLASSIFICATION REPORT:")
print("-"*80)
print(classification_report(y_test, y_pred_final, target_names=['Non-Fraud', 'Fraud']))

# Feature importance (for LightGBM)
if model_name == "Tuned LightGBM":
    print("\n🔍 TOP 20 MOST IMPORTANT FEATURES:")
    print("-"*80)
    feature_importance = pd.DataFrame({
        'feature': X_train_balanced.columns,
        'importance': best_lgb.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    for idx, row in feature_importance.iterrows():
        print(f"   {row['feature']:<45} {row['importance']:>8.1f}")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)

# Save models
import joblib
joblib.dump(best_model, 'fraud_model_best.pkl')
joblib.dump(scaler, 'scaler_robust.pkl')
print(f"\n💾 Best model ({model_name}) saved!")