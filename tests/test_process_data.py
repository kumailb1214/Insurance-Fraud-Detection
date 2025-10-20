import numpy as np
import pandas as pd
import pytest

import app


class DummyScaler:
    def __init__(self, feature_names):
        self.feature_names_in_ = np.array(feature_names)

    def transform(self, X):
        # Return numpy array copy of X values (identity)
        return np.asarray(X)


class DummyModel:
    def predict_proba(self, X):
        # return zeros with shape (n_samples, 2)
        n = X.shape[0]
        return np.zeros((n, 2))


def make_minimal_df(n=3):
    # Data chosen so get_dummies(drop_first=True) produces no columns for DayOfWeek
    df = pd.DataFrame({
        'DayOfWeek': ['Monday'] * n,
        'VehiclePrice': ['30000 to 39000'] * n,
        'Days_Policy_Accident': ['more than 30'] * n,
        'Days_Policy_Claim': ['more than 30'] * n,
        'WeekOfMonth': [1] * n,
        'WeekOfMonthClaimed': [1] * n,
        'Age': [30] * n,
        'PolicyNumber': [100] * n,
        'RepNumber': [5] * n,
        'Deductible': [200] * n,
        'DriverRating': [3] * n,
        'Year': [2000] * n,
    })
    return df


def test_process_data_happy_path(monkeypatch):
    # Prepare dummy artifacts
    scaler_features = [
        'WeekOfMonth', 'WeekOfMonthClaimed', 'Age', 'PolicyNumber', 'RepNumber',
        'Deductible', 'DriverRating', 'Year', 'IsWeekendAccident', 'ClaimDelay',
        'PriceToDeductibleRatio'
    ]

    monkeypatch.setattr(app, 'price_mapping', {'30000 to 39000': 30000})
    monkeypatch.setattr(app, 'days_mapping', {'more than 30': 30})
    monkeypatch.setattr(app, 'days_claim_mapping', {'more than 30': 30})
    monkeypatch.setattr(app, 'target_encoding_maps', {})
    monkeypatch.setattr(app, 'scaler', DummyScaler(scaler_features))
    # model_columns must match the processed columns returned by process_data
    monkeypatch.setattr(app, 'model_columns', list(scaler_features))
    monkeypatch.setattr(app, 'model', DummyModel())

    df = make_minimal_df(n=4)
    # Validate input passes
    ok, missing = app.validate_input_df(df)
    assert ok is True
    assert missing == []

    processed = app.process_data(df)
    # processed should have same number of rows
    assert processed.shape[0] == df.shape[0]
    # columns should include the model_columns
    for col in app.model_columns:
        assert col in processed.columns

    # calling model.predict_proba should work and return vector of length n
    probs = app.model.predict_proba(processed)[:, 1]
    assert probs.shape[0] == df.shape[0]


def test_validate_input_df_missing_columns():
    df = pd.DataFrame({'WeekOfMonth': [1, 2]})
    ok, missing = app.validate_input_df(df)
    assert ok is False
    # DayOfWeek is in required set, so should appear in missing
    assert 'DayOfWeek' in missing
