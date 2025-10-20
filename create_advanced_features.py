import pandas as pd
import numpy as np

df = pd.read_csv('fraud_oracle.csv')

# --- Feature: IsWeekendAccident ---
df['IsWeekendAccident'] = df['DayOfWeek'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# --- Convert Categorical Ranges to Numerical ---

# VehiclePrice
price_mapping = {
    'less than 20000': 10000,
    '20000 to 29000': 25000,
    '30000 to 39000': 35000,
    '40000 to 59000': 50000,
    '60000 to 69000': 65000,
    'more than 69000': 75000
}
df['VehiclePrice_numerical'] = df['VehiclePrice'].map(price_mapping)

# Days_Policy_Accident
days_mapping = {
    'none': 0,
    '1 to 7': 4,
    '8 to 15': 11.5,
    '15 to 30': 22.5,
    'more than 30': 35 # Approximation
}
df['Days_Policy_Accident_numerical'] = df['Days_Policy_Accident'].map(days_mapping)

# Days_Policy_Claim
days_claim_mapping = {
    'none': 0,
    '8 to 15': 11.5,
    '15 to 30': 22.5,
    'more than 30': 35 # Approximation
}
df['Days_Policy_Claim_numerical'] = df['Days_Policy_Claim'].map(days_claim_mapping)


# --- Create New Advanced Features ---

# ClaimDelay
df['ClaimDelay'] = (df['Days_Policy_Claim_numerical'] - df['Days_Policy_Accident_numerical']).abs()

# PriceToDeductibleRatio
df['PriceToDeductibleRatio'] = df['VehiclePrice_numerical'] / (df['Deductible'] + 1e-9)


# --- Save the new dataset ---
df_advanced = df.drop(['VehiclePrice_numerical', 'Days_Policy_Accident_numerical', 'Days_Policy_Claim_numerical'], axis=1)

df_advanced.to_csv('fraud_oracle_advanced_features.csv', index=False)

print("Advanced feature engineering complete.")
new_features = ['IsWeekendAccident', 'ClaimDelay', 'PriceToDeductibleRatio']
print("New features created: ", new_features)
print("New dataset saved as 'fraud_oracle_advanced_features.csv'.")
print("\nPreview of new features:")
print(df_advanced[new_features].head())
