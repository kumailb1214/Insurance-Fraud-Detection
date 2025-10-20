
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('fraud_oracle.csv')

# --- Previous Feature Engineering ---
df['IsWeekendAccident'] = df['DayOfWeek'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

price_mapping = {
    'less than 20000': 10000, '20000 to 29000': 25000, '30000 to 39000': 35000,
    '40000 to 59000': 50000, '60000 to 69000': 65000, 'more than 69000': 75000
}
df['VehiclePrice_numerical'] = df['VehiclePrice'].map(price_mapping)

days_mapping = {
    'none': 0, '1 to 7': 4, '8 to 15': 11.5, '15 to 30': 22.5, 'more than 30': 35
}
df['Days_Policy_Accident_numerical'] = df['Days_Policy_Accident'].map(days_mapping)

days_claim_mapping = {
    'none': 0, '8 to 15': 11.5, '15 to 30': 22.5, 'more than 30': 35
}
df['Days_Policy_Claim_numerical'] = df['Days_Policy_Claim'].map(days_claim_mapping)

df['ClaimDelay'] = (df['Days_Policy_Claim_numerical'] - df['Days_Policy_Accident_numerical']).abs()
df['PriceToDeductibleRatio'] = df['VehiclePrice_numerical'] / (df['Deductible'] + 1e-9)

# --- New Advanced Feature Engineering ---

# 1. Interaction Features
df['Fault_PolicyType_Interaction'] = df['Fault'] + '_' + df['PolicyType']
df['Area_PolicyType_Interaction'] = df['AccidentArea'] + '_' + df['PolicyType']

# 2. Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
numerical_for_poly = df[['ClaimDelay', 'PriceToDeductibleRatio']]
poly_features = poly.fit_transform(numerical_for_poly)
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['ClaimDelay', 'PriceToDeductibleRatio']))

# --- Combine all features ---
df_final = pd.concat([df, poly_df], axis=1)

# Drop intermediate numerical columns
df_final = df_final.drop(['VehiclePrice_numerical', 'Days_Policy_Accident_numerical', 'Days_Policy_Claim_numerical'], axis=1)

# Save the final dataset
df_final.to_csv('fraud_oracle_final_features.csv', index=False)

print("Final feature engineering complete.")
print("New interaction features created: 'Fault_PolicyType_Interaction', 'Area_PolicyType_Interaction'")
print("New polynomial features created for 'ClaimDelay' and 'PriceToDeductibleRatio'.")
print("Final dataset saved as 'fraud_oracle_final_features.csv'.")
