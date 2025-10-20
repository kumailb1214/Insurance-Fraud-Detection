import pandas as pd
import joblib

print("--- Loading Final Model to Extract Feature Importances ---")

# Load the model and the column names
model = joblib.load('fraud_model.joblib')
model_columns = joblib.load('model_columns.joblib')

# Create a dataframe of feature importances
feature_importances = pd.DataFrame({
    'feature': model_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Reset index for clean printing
feature_importances.reset_index(drop=True, inplace=True)

print("\n--- Top 20 Most Important Features for Fraud Detection ---")
print(feature_importances.head(20))