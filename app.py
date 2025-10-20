import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(layout="wide")

# --- Load Model Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('fraud_model.joblib')
        scaler = joblib.load('scaler.joblib')
        model_columns = joblib.load('model_columns.joblib')
        target_encoding_maps = joblib.load('target_encoding_maps.joblib')
        price_mapping = joblib.load('price_mapping.joblib')
        days_mapping = joblib.load('days_mapping.joblib')
        days_claim_mapping = joblib.load('days_claim_mapping.joblib')
        return model, scaler, model_columns, target_encoding_maps, price_mapping, days_mapping, days_claim_mapping
    except FileNotFoundError as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure all .joblib files are in the same directory as the app.")
        return None, None, None, None, None, None, None

model, scaler, model_columns, target_encoding_maps, price_mapping, days_mapping, days_claim_mapping = load_artifacts()

# --- Debugging: Print types of loaded objects ---
st.write(f"Type of model: {type(model)}")
st.write(f"Type of scaler: {type(scaler)}")

# --- Defensive Check for Loaded Artifacts ---
if scaler is not None and not callable(getattr(scaler, 'transform', None)):
    st.error("The 'transform' attribute of the loaded scaler object is not callable. The 'scaler.joblib' file might be corrupt or incorrect.")
    st.stop()

# --- Feature Engineering Function ---
def process_data(df):
    X = df.copy()
    X['IsWeekendAccident'] = X['DayOfWeek'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    X['VehiclePrice_numerical'] = X['VehiclePrice'].map(price_mapping)
    X['Days_Policy_Accident_numerical'] = X['Days_Policy_Accident'].map(days_mapping)
    X['Days_Policy_Claim_numerical'] = X['Days_Policy_Claim'].map(days_claim_mapping)
    X['ClaimDelay'] = (X['Days_Policy_Claim_numerical'] - X['Days_Policy_Accident_numerical']).abs()
    X['PriceToDeductibleRatio'] = X['VehiclePrice_numerical'] / (X['Deductible'] + 1e-9)
    X = X.drop(['VehiclePrice_numerical', 'Days_Policy_Accident_numerical', 'Days_Policy_Claim_numerical'], axis=1)

    for col, mapping in target_encoding_maps.items():
        X[col + '_encoded'] = X[col].map(mapping)
        # mapping may be a dict (where mapping.values() is callable)
        # or a pandas Series (where mapping.values is a numpy.ndarray and not callable).
        try:
            vals = list(mapping.values())
        except TypeError:
            # mapping.values is likely an ndarray (not callable) or similar
            vals = list(mapping.values)
        mean_value = np.mean(vals)
        # Use .loc for safer assignment
        X.loc[:, col + '_encoded'] = X[col + '_encoded'].fillna(mean_value)

    X = X.drop(list(target_encoding_maps.keys()), axis=1)

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    X_cat = pd.get_dummies(X[categorical_features], drop_first=True)

    # Limit numerical features to those the scaler was fitted on (if available)
    try:
        scaler_features = list(getattr(scaler, 'feature_names_in_', []))
    except Exception:
        scaler_features = []

    if scaler_features:
        num_feats_to_use = [f for f in numerical_features if f in scaler_features]
    else:
        num_feats_to_use = list(numerical_features)

    X_num = pd.DataFrame(scaler.transform(X[num_feats_to_use]), columns=num_feats_to_use, index=X.index)
    
    X_processed = pd.concat([X_num, X_cat], axis=1)
    
    missing_cols = set(model_columns) - set(X_processed.columns)
    for c in missing_cols:
        X_processed[c] = 0
    X_processed = X_processed[model_columns]
    
    return X_processed


def get_required_columns():
    """Return the list of columns required by the processing pipeline before one-hot encoding.
    This is used to validate uploaded CSVs and provide helpful errors in the UI.
    """
    # Required raw columns used by process_data
    base_required = [
        'DayOfWeek', 'VehiclePrice', 'Days_Policy_Accident', 'Days_Policy_Claim',
        'Deductible'
    ]
    # Also required columns referenced in target_encoding_maps
    try:
        te_cols = list(target_encoding_maps.keys()) if target_encoding_maps else []
    except Exception:
        te_cols = []

    return list(set(base_required + te_cols))


def validate_input_df(df):
    """Check that uploaded DataFrame contains required columns. Return (ok, missing_columns).
    """
    if df is None:
        return False, ['No data provided']
    required = get_required_columns()
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0), missing

# --- Tiering Function ---
def assign_tier(probability):
    if probability > 0.70:
        return 'Tier 1: URGENT'
    elif probability > 0.20:
        return 'Tier 2: REVIEW'
    else:
        return 'Tier 3: AUTO-APPROVE'

# --- Streamlit UI ---
st.title("Insurance Fraud Detection System")

st.write("""
Upload a CSV file containing insurance claims. The system will predict the probability of fraud for each claim and assign it to an investigation tier.
- **Tier 1 (Urgent):** Probability > 70%
- **Tier 2 (Review):** Probability between 20% and 70%
- **Tier 3 (Auto-Approve):** Probability < 20%
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None and all([model, scaler, model_columns, target_encoding_maps]):
    try:
        input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(input_df.head())

        processed_df = process_data(input_df)
        probabilities = model.predict_proba(processed_df)[:, 1]
        
        results_df = input_df.copy()
        results_df['Fraud Probability'] = probabilities
        results_df['Tier'] = results_df['Fraud Probability'].apply(assign_tier)
        
        st.write("---")
        st.header("Fraud Detection Results")
        
        tier_counts = results_df['Tier'].value_counts()
        st.write("Summary of Tiers:")
        st.dataframe(tier_counts)
        
        st.write("Detailed Results:")
        st.dataframe(results_df)
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(results_df)

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='fraud_detection_results.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")