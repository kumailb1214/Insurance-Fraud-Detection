import joblib
from pathlib import Path

base = Path(__file__).parent
artifacts = [
    'fraud_model.joblib',
    'scaler.joblib',
    'model_columns.joblib',
    'target_encoding_maps.joblib',
    'price_mapping.joblib',
    'days_mapping.joblib',
    'days_claim_mapping.joblib',
]

for name in artifacts:
    path = base / name
    print(f"\n--- {name} ---")
    if not path.exists():
        print("MISSING")
        continue
    obj = joblib.load(path)
    print("Type:", type(obj))
    try:
        # Print small repr
        if hasattr(obj, 'shape'):
            print('shape:', getattr(obj, 'shape'))
        # If it's a dict-like or list-like show keys/len
        try:
            if isinstance(obj, dict):
                print('dict keys sample:', list(obj.keys())[:5])
            elif hasattr(obj, '__len__'):
                print('len:', len(obj))
            else:
                print('repr:', repr(obj)[:200])
        except Exception as e:
            print('Could not show contents:', e)
    except Exception as e:
        print('Error inspecting:', e)

# Also try loading the sample CSV
csv_path = base / 'test_claims.csv'
print('\n--- test_claims.csv ---')
if csv_path.exists():
    import pandas as pd
    df = pd.read_csv(csv_path)
    print('Loaded CSV shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print(df.head().to_string())
else:
    print('MISSING')
