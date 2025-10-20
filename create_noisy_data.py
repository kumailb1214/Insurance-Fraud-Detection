
import pandas as pd
import numpy as np

print("Loading original data to create a noisy test set...")
df = pd.read_csv('fraud_oracle.csv')

# Take a sample
sample_df = df.sample(n=5000, random_state=101)

# --- Add Noise ---
noisy_df = sample_df.copy()

# Identify column types
numerical_cols = noisy_df.select_dtypes(include=np.number).columns.tolist()
if 'FraudFound_P' in numerical_cols:
    numerical_cols.remove('FraudFound_P')
if 'PolicyNumber' in numerical_cols:
    numerical_cols.remove('PolicyNumber')

categorical_cols = noisy_df.select_dtypes(include=['object']).columns.tolist()

# Add noise to numerical columns
noise_factor = 0.05
for col in numerical_cols:
    if noisy_df[col].std() > 0:
        noise = np.random.normal(0, noisy_df[col].std() * noise_factor, noisy_df.shape[0])
        noisy_df[col] = noisy_df[col] + noise
        if 'Age' in col or col in ['Year', 'Deductible', 'RepNumber']:
             noisy_df[col] = noisy_df[col].clip(lower=0)

# Add noise to categorical columns
swap_percentage = 0.05
for col in categorical_cols:
    unique_values = noisy_df[col].unique()
    if len(unique_values) > 1:
        swap_indices = noisy_df.sample(frac=swap_percentage).index
        for i in swap_indices:
            original_value = noisy_df.loc[i, col]
            possible_swaps = [v for v in unique_values if v != original_value]
            if possible_swaps:
                 noisy_df.loc[i, col] = np.random.choice(possible_swaps)

# Save the noisy dataset
noisy_df.to_csv('noisy_test_data.csv', index=False)

print(f"Noisy test file 'noisy_test_data.csv' with {len(noisy_df)} rows created successfully.")
