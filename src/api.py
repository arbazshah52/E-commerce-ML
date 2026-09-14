"""API utilities for loading and using the trained model."""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from .config import META_PATH, MODEL_PATH

logger = logging.getLogger(__name__)


def load_model_package():
    """Load the trained model package from disk.
    
    Returns:
        dict: Contains 'model', 'scaler', and 'model_name'
    
    Raises:
        FileNotFoundError: If model has not been trained yet
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please run src/train.py to train the model first."
        )
    
    logger.info(f"Loading model from {MODEL_PATH}")
    model_package = joblib.load(MODEL_PATH)
    return model_package


def load_metadata() -> dict:
    """Load model metadata.
    
    Returns:
        dict: Metadata about the model including name, metrics, and feature names
    """
    if not META_PATH.exists():
        return {}
    
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    return metadata


def predict_session(
    session_features: dict,
    model_package: dict = None,
) -> dict:
    """Predict whether a session will result in an order.
    
    Args:
        session_features: Dictionary with keys:
            - 'num_clicks': int
            - 'num_carts': int
            - 'num_events': int
            - 'num_unique_items': int
        model_package: Loaded model package (optional, will load if not provided)
    
    Returns:
        dict: Contains 'prediction', 'probability', and 'model_name'
    """
    if model_package is None:
        model_package = load_model_package()
    
    model = model_package["model"]
    scaler = model_package["scaler"]
    model_name = model_package["model_name"]
    
    # Convert features to DataFrame (same format as training)
    feature_names = ["num_clicks", "num_carts", "num_events", "num_unique_items"]
    X = pd.DataFrame([session_features], columns=feature_names)
    
    # Scale if required
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0, 1]  # Probability of order (class 1)
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "model_name": model_name,
        "interpretation": "Order likely" if prediction == 1 else "No order expected",
    }


def get_model_info() -> dict:
    """Get information about the trained model.
    
    Returns:
        dict: Model information including name, performance metrics, and requirements
    """
    metadata = load_metadata()
    
    if not metadata:
        return {"error": "Model not trained yet"}
    
    return {
        "model_name": metadata.get("model_name"),
        "requires_scaling": metadata.get("requires_scaling"),
        "feature_names": metadata.get("feature_names"),
        "test_metrics": metadata.get("test_metrics"),
    }