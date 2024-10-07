# tests/test_model.py
import pytest
from sklearn.linear_model import LinearRegression

def test_model_creation():
    model = LinearRegression()
    assert model is not None

def test_model_training():
    model = LinearRegression()
    # Add test data
    X = [[1, 2], [2, 3], [3, 4]]
    y = [1, 2, 3]
    model.fit(X, y)
    assert model.coef_ is not None
