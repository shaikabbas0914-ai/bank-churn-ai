import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Columns configuration
NUMERICAL_COLS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]
CATEGORICAL_COLS = ["Geography", "Gender"]
TARGET_COL = "Exited"
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

def get_preprocessor():
    """
    Creates the scikit-learn column transformer preprocessor.
    """
    try:
        # In newer scikit-learn versions, sparse_output is used instead of sparse
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", encoder, CATEGORICAL_COLS)
        ],
        remainder="drop" # drops columns not specified in transformers
    )
    return preprocessor

def load_and_preprocess_data(csv_path="data/Churn_Modelling.csv", test_size=0.2, random_state=42):
    """
    Loads raw CSV, performs train-test split, fits the preprocessor,
    saves the preprocessor, and returns the preprocessed splits.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run download script first.")

    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Split into features and target
    X = df.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
    y = df[TARGET_COL]
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # Create and fit preprocessor
    preprocessor = get_preprocessor()
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the fitted preprocessor
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor successfully fitted and saved to: {preprocessor_path}")
    
    # Get feature names after transformation for reference
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    feature_names = NUMERICAL_COLS + cat_feature_names
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor, features = load_and_preprocess_data()
    print("Preprocessing completed!")
    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")
    print(f"Features: {features}")
