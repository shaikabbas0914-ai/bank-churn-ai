import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import tensorflow as tf

# Target and Dropped columns
TARGET_COL = "Exited"
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

def evaluate_models():
    csv_path = "data/Churn_Modelling.csv"
    preprocessor_path = "models/preprocessor.pkl"
    comparison_path = "models/model_comparison.json"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run download script first.")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Please run training script first.")
        
    print("Loading test data...")
    df = pd.read_csv(csv_path)
    X = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    y = df[TARGET_COL]
    
    # Train-test split (recreating the test split identically using the same seed and stratification)
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Load preprocessor and transform test set
    preprocessor = joblib.load(preprocessor_path)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Test data shape: {X_test_processed.shape}")
    
    # Model files
    models_info = {
        "Logistic Regression": ("models/logistic_regression.pkl", "sklearn"),
        "Random Forest": ("models/random_forest.pkl", "sklearn"),
        "XGBoost": ("models/xgboost.pkl", "sklearn"),
        "ANN": ("models/churn_model.keras", "keras")
    }
    
    print("\n" + "="*50)
    print("                 MODEL EVALUATION REPORT")
    print("="*50)
    
    for model_name, (model_path, model_type) in models_info.items():
        if not os.path.exists(model_path):
            print(f"\nModel file not found for {model_name} at {model_path}. Skipping.")
            continue
            
        print(f"\nEvaluating {model_name}...")
        
        if model_type == "keras":
            model = tf.keras.models.load_model(model_path)
            y_prob = model.predict(X_test_processed).flatten()
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            model = joblib.load(model_path)
            y_prob = model.predict_proba(X_test_processed)[:, 1]
            y_pred = model.predict(X_test_processed)
            
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Stayed (0)", "Churned (1)"]))
        
        print(f"{model_name} Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"  True Negatives (Stayed correctly predicted)  : {cm[0][0]}")
        print(f"  False Positives (Predicted churn, stayed)     : {cm[0][1]}")
        print(f"  False Negatives (Predicted stay, churned)     : {cm[1][0]}")
        print(f"  True Positives (Churned correctly predicted) : {cm[1][1]}")
        
        # Print summary line of metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        print(f"\nSummary - Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print("-" * 50)
        
    if os.path.exists(comparison_path):
        print("\nStored Comparison Metrics (JSON):")
        with open(comparison_path, "r") as f:
            metrics = json.load(f)
            print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    evaluate_models()
