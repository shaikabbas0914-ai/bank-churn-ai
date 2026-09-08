import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import preprocessing utility
from data_preprocessing import load_and_preprocess_data

def train_all_models():
    # 1. Load preprocessed data
    print("Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = load_and_preprocess_data()
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Dictionary to hold metrics for comparison
    model_metrics = {}
    
    # 2. Compute Class Weights for ANN and other models
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(classes[i]): float(weights[i]) for i in range(len(classes))}
    print(f"Calculated class weights: {class_weight_dict}")
    
    # 3. Train Comparison Model: Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, "models/logistic_regression.pkl")
    print("Saved Logistic Regression to models/logistic_regression.pkl")
    
    # 4. Train Comparison Model: Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "models/random_forest.pkl")
    print("Saved Random Forest to models/random_forest.pkl")
    
    # 5. Train Comparison Model: XGBoost
    print("\nTraining XGBoost...")
    # Calculate scale_pos_weight for XGBoost to handle imbalance
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = float(num_neg) / float(num_pos)
    
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_estimators=100
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, "models/xgboost.pkl")
    print("Saved XGBoost to models/xgboost.pkl")
    
    # 6. Build and Train Deep Learning Model: ANN
    print("\nBuilding and Training ANN...")
    
    ann_model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    
    # Compile the model
    ann_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        filepath="models/churn_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    # Fit the ANN model
    history = ann_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Save in multiple formats for maximum cross-platform / cross-version compatibility
    try:
        ann_model.save("models/churn_model.keras")
    except Exception as e:
        print(f"Note on keras format save: {e}")
    try:
        ann_model.save_weights("models/churn_model.weights.h5")
        print("Saved ANN weights to models/churn_model.weights.h5")
    except Exception as e:
        print(f"Note on weights save: {e}")
    try:
        ann_model.save("models/churn_model.h5")
        print("Saved ANN model to models/churn_model.h5")
    except Exception as e:
        print(f"Note on h5 save: {e}")
    
    # 7. Model Evaluation
    print("\nEvaluating all models on test data...")
    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "ANN": ann_model
    }
    
    for model_name, model in models.items():
        if model_name == "ANN":
            # ANN output is probability of positive class
            y_prob = model.predict(X_test).flatten()
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        model_metrics[model_name] = {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1),
            "ROC-AUC": float(auc)
        }
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        
    # Save the metrics to a json file
    comparison_path = "models/model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(model_metrics, f, indent=4)
    print(f"\nModel metrics saved to: {comparison_path}")

if __name__ == "__main__":
    train_all_models()
