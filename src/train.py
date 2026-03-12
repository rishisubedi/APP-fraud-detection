import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, classification_report
import os
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load the generated transaction data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Please run src/data_generator.py first.")
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess data for modeling, dropping identifiers and extracting target."""
    logger.info("Preprocessing data...")
    # Drop identifiers and timestamp since they are not predictive features
    drop_cols = ['transaction_id', 'account_id', 'receiver_account_id', 'timestamp']
    
    # Ensure columns exist before dropping to avoid KeyError
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    X = df.drop(columns=drop_cols + ['is_fraud'])
    y = df['is_fraud']
    
    logger.info(f"Features used for training: {list(X.columns)}")
    return X, y

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """Train XGBoost with cost-sensitive learning to handle class imbalance."""
    logger.info("Training XGBoost Classifier...")
    
    # Calculate scale_pos_weight for highly imbalanced class
    neg_cases = (y_train == 0).sum()
    pos_cases = (y_train == 1).sum()
    scale_pos_weight = neg_cases / pos_cases
    logger.info(f"Class imbalance handling (neg/pos): scale_pos_weight={scale_pos_weight:.2f}")

    # Initialize model with scale_pos_weight
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Evaluate the model using precision-recall AUC and F1-score."""
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Precision-Recall AUC: {pr_auc:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    
    # Optional full classification report for details
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])
    logger.info(f"\nClassification Report:\n{report}")

def save_model(model: xgb.XGBClassifier, output_dir: str = 'models', filename: str = 'xgb_fraud_model.json') -> None:
    """Save the trained model as an artifact."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, filename)
    model.save_model(model_path)
    logger.info(f"Model saved successfully to {model_path}")

def main():
    try:
        # Load dataset
        data_path = os.path.join('data', 'transactions.csv')
        df = load_data(data_path)
        
        # Preprocess
        X, y = preprocess_data(df)
        
        # Train-test split (80% train, 20% test, stratified to maintain fraud ratio)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Train
        model = train_model(X_train, y_train)
        
        # Evaluate
        evaluate_model(model, X_test, y_test)
        
        # Save Artifact
        save_model(model)
        
    except Exception as e:
        logger.error(f"Error during training pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    main()
