import json
import warnings
import joblib
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from config import MODEL_PATH, META_PATH, MODEL_DIR


# 1. Load data

def load_data(data_path):
    print("Loading data..")
    df = pd.read_csv(data_path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    return df


# 2. Create features and target

def create_features_and_target(df):
    print("Creating features and target...")
    # Create a copy so we do not change the original dataframe
    df = df.copy()

    # Create target for every session
    session_target = df.groupby("session")["type"].max()
    # Convert target to 1 or 0
    session_target = session_target.apply(check_if_order).astype(int)

    # Create separate columns for each event
    df["is_click"] = (df["type"] == "clicks").astype(int)
    df["is_cart"] = (df["type"] == "carts").astype(int)

    # Create session-level features
    session_features = df.groupby("session").agg(
        num_clicks=("is_click", "sum"),
        num_carts=("is_cart", "sum"),
        num_events=("type", "count"),
        num_unique_items=("aid", "nunique"),
    )

    # Add target
    session_features["target"] = session_target

    # X = features
    x = session_features.drop(columns=["target"])
    # y = target
    y = session_features["target"]

    print("Number of sessions:", len(x))
    print("Number of features:", x.shape[1])
    print("Target distribution:")
    print(y.value_counts())
    return x, y


def check_if_order(value):
    if value == "orders":
        return True
    else:
        return False


# 3. Split data into train and test:
# 80% train + validation, 20% test

def split_data(x, y):
    print("Splitting data into train and test..")
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Number of training samples:", len(x_train_val))
    print("Number of test samples:", len(x_test))

    # 85% of train_val = train, 15% of train_val = validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.15, random_state=40, stratify=y_train_val,
    )
    print("Train rows:", len(x_train))
    print("Validation rows:", len(x_val))
    print("Test rows:", len(x_test))
    return (x_train, x_val, x_test, y_train, y_val, y_test, x_train_val, y_train_val)


# 4. Scale data

def scale_data(x_train, x_val, x_test):
    print("Scaling data..")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    # Use the same scaler for validation and test
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    return (x_train_scaled, x_val_scaled, x_test_scaled, scaler)


# Evaluate model

def evaluate_model(model_name, y_true, predictions, probabilities):
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
    }


# 5. Model training and hyperparameter tuning

def train_models(x_train_scaled, x_train, x_val_scaled, x_val, y_train, y_val):
    print("Training models..")
    models_data = {}

    # Logistic Regression
    print("Training Logistic Regression...")
    log_reg = LogisticRegression(class_weight="balanced", random_state=42)
    log_reg_params = {
        "C": [0.01, 0.1, 1, 10, 100],
        "l1_ratio": [0, 1],
        "solver": ["liblinear"],
        "max_iter": [500, 1000],
    }
    log_grid = GridSearchCV(
        estimator=log_reg, param_grid=log_reg_params, scoring="f1", cv=5, n_jobs=-1,
    )
    log_grid.fit(x_train_scaled, y_train)
    best_log_reg = log_grid.best_estimator_
    log_val_pred = best_log_reg.predict(x_val_scaled)
    log_val_prob = best_log_reg.predict_proba(x_val_scaled)[:, 1]
    print("Best Logistic Regression parameters:", log_grid.best_params_)
    models_data["logreg"] = {
        "model": best_log_reg, "scaler": True,
        "predictions": log_val_pred, "probabilities": log_val_prob,
    }

    # SVC
    print("Training SVC...")
    svc = SVC(class_weight="balanced", probability=True, random_state=42)
    svc_params = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }
    svc_grid = GridSearchCV(
        estimator=svc, param_grid=svc_params, scoring="f1", cv=5, n_jobs=-1,
    )
    svc_grid.fit(x_train_scaled, y_train)
    best_svc_base = svc_grid.best_estimator_
    # Calibrate probabilities
    best_svc = CalibratedClassifierCV(estimator=best_svc_base, method="sigmoid", cv=5)
    best_svc.fit(x_train_scaled, y_train)
    svc_val_pred = best_svc.predict(x_val_scaled)
    svc_val_prob = best_svc.predict_proba(x_val_scaled)[:, 1]
    print("Best SVC parameters:", svc_grid.best_params_)
    models_data["svc"] = {
        "model": best_svc, "scaler": True,
        "predictions": svc_val_pred, "probabilities": svc_val_prob,
    }

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    rf_params = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf_grid = GridSearchCV(
        estimator=rf, param_grid=rf_params, scoring="f1", cv=5, n_jobs=-1,
    )
    rf_grid.fit(x_train, y_train)
    best_rf = rf_grid.best_estimator_
    rf_val_pred = best_rf.predict(x_val)
    rf_val_prob = best_rf.predict_proba(x_val)[:, 1]
    print("Best Random Forest parameters:", rf_grid.best_params_)
    models_data["rf"] = {
        "model": best_rf, "scaler": False,
        "predictions": rf_val_pred, "probabilities": rf_val_prob,
    }

    # Compare models
    results = pd.DataFrame([
        evaluate_model("Logistic Regression", y_val, log_val_pred, log_val_prob),
        evaluate_model("SVC", y_val, svc_val_pred, svc_val_prob),
        evaluate_model("Random Forest", y_val, rf_val_pred, rf_val_prob),
    ])
    results = results.sort_values("f1_score", ascending=False)
    print("\nValidation results:")
    print(results.to_string(index=False))
    return models_data, results


# 6. Select best model and retrain on train+val

def select_and_retrain_best_model(
    models_data, results, x_train, x_val, x_test, y_train, y_val, y_test, x_train_val, y_train_val,
):
    print("Selecting best model...")
    # First row has the highest F1 score
    best_model_name = results.iloc[0]["model"]
    print("Best model:", best_model_name)

    if best_model_name == "Logistic Regression":
        print("Retraining Logistic Regression...")
        final_scaler = StandardScaler()
        x_train_val_final = final_scaler.fit_transform(x_train_val)
        x_test_final = final_scaler.transform(x_test)
        best_model = models_data["logreg"]["model"]
    elif best_model_name == "SVC":
        print("Retraining SVC...")
        final_scaler = StandardScaler()
        x_train_val_final = final_scaler.fit_transform(x_train_val)
        x_test_final = final_scaler.transform(x_test)
        best_model = models_data["svc"]["model"]
    else:
        print("Retraining Random Forest...")
        final_scaler = None
        x_train_val_final = x_train_val
        x_test_final = x_test
        best_model = models_data["rf"]["model"]

    # Train on train + validation data (runs no matter which model was picked)
    best_model.fit(x_train_val_final, y_train_val)

    # Make test predictions
    best_test_pred = best_model.predict(x_test_final)
    best_test_prob = best_model.predict_proba(x_test_final)[:, 1]

    # Calculate test metrics
    test_metrics = {
        "accuracy": accuracy_score(y_test, best_test_pred),
        "precision": precision_score(y_test, best_test_pred, zero_division=0),
        "recall": recall_score(y_test, best_test_pred, zero_division=0),
        "f1_score": f1_score(y_test, best_test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, best_test_prob),
        "average_precision": average_precision_score(y_test, best_test_prob),
    }

    # Print classification report
    print("\nClassification report:")
    print(classification_report(
        y_test, best_test_pred, target_names=["No order", "Order"], zero_division=0,
    ))

    # Print confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_test, best_test_pred))

    # Print metrics
    print("\nTest metrics:")
    for metric, value in test_metrics.items():
        print(metric, ":", round(value, 4))

    return (best_model, final_scaler, best_model_name, test_metrics)


# Save model

def save_model(model, scaler, model_name, test_metrics):
    print("Saving model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Put model and scaler together
    model_package = {"model": model, "scaler": scaler, "model_name": model_name}

    # Save model
    joblib.dump(model_package, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

    # Create metadata
    metadata = {
        "model_name": model_name,
        "requires_scaling": scaler is not None,
        "test_metrics": test_metrics,
        "feature_names": ["num_clicks", "num_carts", "num_events", "num_unique_items"],
    }

    # Save metadata
    metadata_path = MODEL_DIR / "metadata.json"
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2)
    print("Metadata saved to:", metadata_path)


# 7. Complete training pipeline

def train_pipeline(data_path=None):
    if data_path is None:
        data_path = (
            Path(__file__).resolve().parents[1]
            / "data" / "events_10000_sessions.csv"
        )

    print("Starting training pipeline")
    print("-------------------------")

    # Step 1: Load data
    df = load_data(data_path)

    # Step 2: Create features
    x, y = create_features_and_target(df)

    # Step 3: Split data
    (x_train, x_val, x_test, y_train, y_val, y_test, x_train_val, y_train_val) = split_data(x, y)

    # Step 4: Scale data
    (x_train_scaled, x_val_scaled, x_test_scaled, scaler) = scale_data(x_train, x_val, x_test)

    # Step 5: Train models
    models_data, results = train_models(
        x_train_scaled, x_train, x_val_scaled, x_val, y_train, y_val,
    )

    # Step 6: Select best model
    (best_model, final_scaler, model_name, test_metrics) = select_and_retrain_best_model(
        models_data, results, x_train, x_val, x_test, y_train, y_val, y_test, x_train_val, y_train_val,
    )

    # Step 7: Save model
    save_model(best_model, final_scaler, model_name, test_metrics)
    print("-------------------------")
    print("Training completed!")
    print("Best model:", model_name)
    return best_model, final_scaler


# Run training
if __name__ == "__main__":
    train_pipeline()