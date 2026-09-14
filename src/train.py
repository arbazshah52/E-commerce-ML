"""E-commerce ML training pipeline.

This script trains multiple classification models to predict whether a customer
session will result in an order. The best model (based on validation F1-score)
is saved using joblib for later use in the Streamlit app.
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import MODEL_DIR, MODEL_PATH

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Load event-level data from CSV."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    return df


def create_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create session-level features and target variable.
    
    Features:
    - num_clicks: Count of click events in session
    - num_carts: Count of cart events in session
    - num_events: Total events in session
    - num_unique_items: Unique items viewed in session
    
    Target:
    - 1 if session contains at least one order, 0 otherwise
    """
    logger.info("Creating session-level features and target")

    # Create session target: 1 if "orders" in session, 0 otherwise
    session_target = (
        df.groupby("session")["type"]
        .apply(lambda values: ("orders" in values.values))
        .astype(int)
    )

    # Aggregate event-level data to session level
    session_features = df.groupby("session").agg(
        num_clicks=("type", lambda x: (x == "clicks").sum()),
        num_carts=("type", lambda x: (x == "carts").sum()),
        num_events=("type", "count"),
        num_unique_items=("aid", "nunique"),
    )

    session_features["target"] = session_target

    x = session_features.drop(columns=["target"])
    y = session_features["target"]

    logger.info(f"Created {len(x)} sessions with {x.shape[1]} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    return x, y


def split_data(
    x: pd.DataFrame, y: pd.Series, train_size: float = 0.8, val_size: float = 0.15
) -> tuple:
    """Split data into train, validation, and test sets."""
    logger.info(f"Splitting data: train={train_size}, val={val_size}")

    # First split: 80% train+val, 20% test
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=1 - train_size, random_state=42, stratify=y
    )

    # Second split: split train+val into 85% train, 15% val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=val_size, random_state=40, stratify=y_train_val
    )

    logger.info(
        f"Data split - Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}"
    )

    return x_train, x_val, x_test, y_train, y_val, y_test, x_train_val, y_train_val


def scale_data(x_train, x_val, x_test):
    """Fit scaler on training data and transform all sets."""
    logger.info("Scaling features using StandardScaler")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_val_scaled, x_test_scaled, scaler


def evaluate_model(model_name: str, y_true: pd.Series, predictions, probabilities):
    """Calculate evaluation metrics for a model."""
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
    }


def train_models(x_train_scaled, x_train, x_val_scaled, x_val, y_train, y_val):
    """Train three classification models with GridSearchCV.
    
    Returns:
        - Dictionary with trained models and their predictions
        - DataFrame with validation performance metrics
    """
    logger.info("Training models with GridSearchCV")

    models_data = {}

    # ---- Logistic Regression ----
    logger.info("Training Logistic Regression...")
    log_reg_base = LogisticRegression(class_weight="balanced", random_state=42)
    param_grid_logreg = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "lbfgs"],
        "max_iter": [500, 1000, 2000],
    }
    grid_log = GridSearchCV(
        estimator=log_reg_base,
        param_grid=param_grid_logreg,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    grid_log.fit(x_train_scaled, y_train)
    best_log_reg = grid_log.best_estimator_
    log_val_pred = best_log_reg.predict(x_val_scaled)
    log_val_prob = best_log_reg.predict_proba(x_val_scaled)[:, 1]
    logger.info(f"Logistic Regression best params: {grid_log.best_params_}")

    models_data["logreg"] = {
        "model": best_log_reg,
        "scaler": True,  # Requires scaling
        "predictions": log_val_pred,
        "probabilities": log_val_prob,
    }

    # ---- Support Vector Classifier ----
    logger.info("Training Support Vector Classifier...")
    svc_base = SVC(class_weight="balanced", probability=True, random_state=42)
    param_grid_svc = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }
    grid_svc = GridSearchCV(
        estimator=svc_base,
        param_grid=param_grid_svc,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    grid_svc.fit(x_train_scaled, y_train)
    best_svc_base = grid_svc.best_estimator_

    # Calibrate SVC predictions
    best_svc = CalibratedClassifierCV(
        estimator=best_svc_base,
        method="sigmoid",
        cv=5,
    )
    best_svc.fit(x_train_scaled, y_train)
    svc_val_pred = best_svc.predict(x_val_scaled)
    svc_val_prob = best_svc.predict_proba(x_val_scaled)[:, 1]
    logger.info(f"SVC best params: {grid_svc.best_params_}")

    models_data["svc"] = {
        "model": best_svc,
        "scaler": True,  # Requires scaling
        "predictions": svc_val_pred,
        "probabilities": svc_val_prob,
    }

    # ---- Random Forest ----
    logger.info("Training Random Forest...")
    rf_base = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    param_grid_rf = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    grid_rf = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid_rf,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    grid_rf.fit(x_train, y_train)
    best_rf = grid_rf.best_estimator_
    rf_val_pred = best_rf.predict(x_val)
    rf_val_prob = best_rf.predict_proba(x_val)[:, 1]
    logger.info(f"Random Forest best params: {grid_rf.best_params_}")

    models_data["rf"] = {
        "model": best_rf,
        "scaler": False,  # Does not require scaling
        "predictions": rf_val_pred,
        "probabilities": rf_val_prob,
    }

    # Compare validation performance
    results = pd.DataFrame(
        [
            evaluate_model("Logistic Regression", y_val, log_val_pred, log_val_prob),
            evaluate_model("SVC", y_val, svc_val_pred, svc_val_prob),
            evaluate_model("Random Forest", y_val, rf_val_pred, rf_val_prob),
        ]
    ).sort_values("f1_score", ascending=False)

    logger.info("\nValidation Performance:\n" + str(results.to_string(index=False)))

    return models_data, results


def select_and_retrain_best_model(
    models_data, results, x_train, x_val, x_test, y_train, y_val, y_test, x_train_val, y_train_val
):
    """Select best model based on validation F1-score and retrain on train+val data.
    
    Returns:
        - Best model
        - Scaler (if applicable)
        - Test metrics dictionary
    """
    best_model_name = results.iloc[0]["model"]
    logger.info(f"\nBest model (validation F1-score): {best_model_name}")

    # Refit the selected model on the combined training and validation data
    if best_model_name == "Logistic Regression":
        logger.info("Retraining Logistic Regression on train+val data...")
        final_scaler = StandardScaler()
        x_train_val_final = final_scaler.fit_transform(x_train_val)
        x_test_final = final_scaler.transform(x_test)
        best_model = models_data["logreg"]["model"]

    elif best_model_name == "SVC":
        logger.info("Retraining SVC on train+val data...")
        final_scaler = StandardScaler()
        x_train_val_final = final_scaler.fit_transform(x_train_val)
        x_test_final = final_scaler.transform(x_test)
        best_model = models_data["svc"]["model"]

    else:  # Random Forest
        logger.info("Retraining Random Forest on train+val data...")
        final_scaler = None
        x_train_val_final = x_train_val
        x_test_final = x_test
        best_model = models_data["rf"]["model"]

    # Fit the best model on combined train+val data
    best_model.fit(x_train_val_final, y_train_val)

    # Evaluate on test set
    best_test_pred = best_model.predict(x_test_final)
    best_test_prob = best_model.predict_proba(x_test_final)[:, 1]

    test_metrics = {
        "accuracy": accuracy_score(y_test, best_test_pred),
        "precision": precision_score(y_test, best_test_pred, zero_division=0),
        "recall": recall_score(y_test, best_test_pred, zero_division=0),
        "f1_score": f1_score(y_test, best_test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, best_test_prob),
        "average_precision": average_precision_score(y_test, best_test_prob),
    }

    logger.info("\nTest Set Classification Report:")
    logger.info(
        classification_report(
            y_test,
            best_test_pred,
            target_names=["No order", "Order"],
            zero_division=0,
        )
    )

    logger.info("Confusion matrix:")
    logger.info(str(confusion_matrix(y_test, best_test_pred)))

    logger.info("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return best_model, final_scaler, best_model_name, test_metrics


def save_model(model, scaler, model_name: str, test_metrics: dict):
    """Save best model and scaler to disk using joblib.
    
    Also saves metadata about the model and its performance.
    """
    logger.info(f"\nSaving model and metadata to {MODEL_PATH}...")

    # Create model directory if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Create model package: dict with model, scaler, and name
    model_package = {
        "model": model,
        "scaler": scaler,
        "model_name": model_name,
    }

    # Save model package
    joblib.dump(model_package, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "requires_scaling": scaler is not None,
        "test_metrics": test_metrics,
        "feature_names": ["num_clicks", "num_carts", "num_events", "num_unique_items"],
    }

    metadata_path = MODEL_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")


def train_pipeline(data_path: str | Path = None):
    """Complete training pipeline.
    
    Args:
        data_path: Path to events CSV. If None, looks for data/events_10000_sessions.csv
    """
    if data_path is None:
        data_path = Path(__file__).resolve().parents[1] / "data" / "events_10000_sessions.csv"

    logger.info("Starting E-commerce ML Training Pipeline")
    logger.info("=" * 60)

    # Load data
    df = load_data(data_path)

    # Feature engineering
    x, y = create_features_and_target(df)

    # Split data
    x_train, x_val, x_test, y_train, y_val, y_test, x_train_val, y_train_val = split_data(x, y)

    # Scale data
    x_train_scaled, x_val_scaled, x_test_scaled, scaler = scale_data(x_train, x_val, x_test)

    # Train models
    models_data, results = train_models(
        x_train_scaled, x_train, x_val_scaled, x_val, y_train, y_val
    )

    # Select best model and retrain on train+val
    best_model, final_scaler, model_name, test_metrics = select_and_retrain_best_model(
        models_data, results, x_train, x_val, x_test, y_train, y_val, y_test,
        x_train_val, y_train_val
    )

    # Save model
    save_model(best_model, final_scaler, model_name, test_metrics)

    logger.info("=" * 60)
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Best model saved: {model_name}")

    return best_model, final_scaler


if __name__ == "__main__":
    train_pipeline()
