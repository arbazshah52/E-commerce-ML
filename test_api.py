"""Test the API functions to verify model loading and prediction."""

from src.api import load_model_package, predict_session, get_model_info

print("=" * 60)
print("Testing Model API Functions")
print("=" * 60)

# Get model info
info = get_model_info()
print("\n✓ Model Info:")
print(f'  Model: {info["model_name"]}')
print(f'  Requires Scaling: {info["requires_scaling"]}')
print(f'  Features: {info["feature_names"]}')

# Load model
model_package = load_model_package()
print("\n✓ Model loaded successfully")

# Make a prediction
test_session = {
    "num_clicks": 15,
    "num_carts": 3,
    "num_events": 20,
    "num_unique_items": 8,
}

result = predict_session(test_session, model_package)
print("\n✓ Prediction Test:")
print(f'  Input: {test_session}')
print(f'  Prediction: {result["prediction"]} ({result["interpretation"]})')
print(f'  Probability: {result["probability"]:.4f}')

print("\n" + "=" * 60)
print("All API functions working correctly!")
print("=" * 60)
