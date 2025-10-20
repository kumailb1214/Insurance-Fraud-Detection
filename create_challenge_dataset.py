
import pandas as pd
import numpy as np

print("Generating the final challenge dataset...")

# Load original data to sample from for base values
df = pd.read_csv('fraud_oracle.csv')

# --- 1. Define Profiles ---

high_risk_profile = {
    'Month': 'Mar', 'WeekOfMonth': 1, 'DayOfWeek': 'Friday', 'Make': 'Honda', 'AccidentArea': 'Rural',
    'DayOfWeekClaimed': 'Monday', 'MonthClaimed': 'Mar', 'WeekOfMonthClaimed': 2, 'Sex': 'Male',
    'MaritalStatus': 'Single', 'Age': 32, 'Fault': 'Policy Holder', 'PolicyType': 'Sedan - All Perils',
    'VehicleCategory': 'Sedan', 'VehiclePrice': 'more than 69000', 'RepNumber': 12, 'Deductible': 400,
    'DriverRating': 1, 'Days_Policy_Accident': 'more than 30', 'Days_Policy_Claim': 'more than 30',
    'PastNumberOfClaims': 'none', 'AgeOfVehicle': 'new', 'AgeOfPolicyHolder': '31 to 35',
    'PoliceReportFiled': 'No', 'WitnessPresent': 'No', 'AgentType': 'External', 'NumberOfSuppliments': 'more than 5',
    'AddressChange_Claim': '2 to 3 years', 'NumberOfCars': '1 vehicle', 'Year': 1995, 'BasePolicy': 'All Perils',
    'FraudFound_P': 1
}

low_risk_profile = {
    'Month': 'Jul', 'WeekOfMonth': 3, 'DayOfWeek': 'Tuesday', 'Make': 'Toyota', 'AccidentArea': 'Urban',
    'DayOfWeekClaimed': 'Tuesday', 'MonthClaimed': 'Jul', 'WeekOfMonthClaimed': 3, 'Sex': 'Female',
    'MaritalStatus': 'Married', 'Age': 55, 'Fault': 'Third Party', 'PolicyType': 'Sedan - Liability',
    'VehicleCategory': 'Sedan', 'VehiclePrice': '20000 to 29000', 'RepNumber': 8, 'Deductible': 400,
    'DriverRating': 4, 'Days_Policy_Accident': 'more than 30', 'Days_Policy_Claim': 'more than 30',
    'PastNumberOfClaims': '2 to 4', 'AgeOfVehicle': '7 years', 'AgeOfPolicyHolder': '51 to 65',
    'PoliceReportFiled': 'Yes', 'WitnessPresent': 'Yes', 'AgentType': 'Internal', 'NumberOfSuppliments': 'none',
    'AddressChange_Claim': 'no change', 'NumberOfCars': '1 vehicle', 'Year': 1996, 'BasePolicy': 'Liability',
    'FraudFound_P': 0
}

# --- 2. Generate Data ---

high_risk_claims = pd.DataFrame([high_risk_profile] * 15)
low_risk_claims = pd.DataFrame([low_risk_profile] * 50)

# Add some minor variation to the generated claims
for i in range(len(high_risk_claims)):
    high_risk_claims.loc[i, 'Age'] = high_risk_claims.loc[i, 'Age'] + np.random.randint(-3, 3)
for i in range(len(low_risk_claims)):
    low_risk_claims.loc[i, 'Age'] = low_risk_claims.loc[i, 'Age'] + np.random.randint(-5, 5)

# --- 3. Combine with Normal Data ---

# Take a random sample of normal claims
normal_claims = df.sample(n=50, random_state=202)

# Combine all parts
challenge_df = pd.concat([high_risk_claims, low_risk_claims, normal_claims], ignore_index=True)

# Add unique PolicyNumbers
challenge_df['PolicyNumber'] = range(len(challenge_df))

# Shuffle the final dataset
challenge_df = challenge_df.sample(frac=1, random_state=1).reset_index(drop=True)

# Save to CSV
challenge_df.to_csv('challenge_dataset.csv', index=False)

print(f"Final challenge dataset 'challenge_dataset.csv' with {len(challenge_df)} rows created successfully.")
