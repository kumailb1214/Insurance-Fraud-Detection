import joblib
s = joblib.load('scaler.joblib')
print('scaler type', type(s))
print('has feature_names_in_?', hasattr(s, 'feature_names_in_'))
if hasattr(s, 'feature_names_in_'):
    print('feature_names_in_:', list(s.feature_names_in_))
else:
    print('No feature_names_in_ attribute')
