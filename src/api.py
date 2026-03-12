from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state for ML model and SHAP explainer
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_fraud_model.json")
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI to load ML models efficiently on startup."""
    logger.info(f"Loading model from {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        ml_models["model"] = model
        logger.info("Model loaded successfully.")
        
        logger.info("Initializing SHAP explainer...")
        # TreeExplainer is heavily optimized for XGBoost
        ml_models["explainer"] = shap.TreeExplainer(model)
        logger.info("SHAP explainer initialized.")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}. Ensure the model is trained before hitting /predict.")
    
    yield  # Server runs while yielding
    
    # Cleanup resources on shutdown
    ml_models.clear()
    logger.info("Cleaned up ML models.")

app = FastAPI(
    title="APP Fraud Detection API",
    description="Real-time Authorised Push Payment Fraud Detection API with SHAP Explanations for FCA regulatory compliance.",
    version="1.0.0",
    lifespan=lifespan
)

FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.80")) # Business logic threshold to block transactions

# Pydantic models for request validation
class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique UUID for the transaction")
    account_id: str = Field(..., description="Sender's account ID")
    receiver_account_id: str = Field(..., description="Receiver's account ID")
    amount_gbp: float = Field(..., ge=0, description="Transaction amount in British Pounds (GBP)")
    is_new_payee: bool = Field(..., description="True if the receiver is a new payee, False otherwise")
    device_risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100) from device telemetry")

class FraudResponse(BaseModel):
    transaction_id: str
    fraud_probability_score: float
    block_transaction: bool
    top_reasons: Dict[str, float]

@app.post("/predict", response_model=FraudResponse)
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict the probability of APP fraud for a given transaction and explain the decision.
    """
    model = ml_models.get("model")
    explainer = ml_models.get("explainer")
    
    if model is None or explainer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is currently unavailable. Please ensure the model file is accessible and restart the server."
        )

    # Note: Column order MUST exactly match the order used during training in train.py
    feature_names = ['amount_gbp', 'is_new_payee', 'device_risk_score']
    
    # Convert bool to int for XGBoost
    features_dict = {
        'amount_gbp': [transaction.amount_gbp],
        'is_new_payee': [1 if transaction.is_new_payee else 0],
        'device_risk_score': [transaction.device_risk_score]
    }
    
    input_df = pd.DataFrame(features_dict, columns=feature_names)
    
    try:
        # 1. Predict Fraud Probability
        # predict_proba returns a 2D array, we want the probability of class 1 (Fraud)
        prob = model.predict_proba(input_df)[0, 1]
        
        # 2. Extract SHAP Explanations for Compliance
        # shap_values represents how much each feature pushes the prediction log-odds
        shap_values = explainer.shap_values(input_df)
        
        # For Binary Classification, shap_values might be 1D or 2D depending on xgboost version/objective
        # Typically for binary logistic, it's 1D per instance.
        instance_shap = shap_values[0] if len(shap_values.shape) == 2 else shap_values
        
        reasons = {name: float(val) for name, val in zip(feature_names, instance_shap)}
        
        # We define "top reasons" as the features that strongly pushed the model towards FRAUD (positive SHAP).
        # We'll return the positive contributors sorted by impact.
        top_reasons = {k: v for k, v in sorted(reasons.items(), key=lambda item: item[1], reverse=True) if v > 0}
        
        # If there are no positive contributors (very clean transaction), return the most important mitigating factors
        if not top_reasons:
            top_reasons = {k: v for k, v in sorted(reasons.items(), key=lambda item: abs(item[1]), reverse=True)}

        # 3. Apply Decision Engine Rule
        block = bool(prob >= FRAUD_THRESHOLD)
        
        logger.info(f"Transaction {transaction.transaction_id} processed: score={prob:.3f}, block={block}")
        
        return FraudResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability_score=float(prob),
            block_transaction=block,
            top_reasons=top_reasons
        )
            
    except Exception as e:
        logger.error(f"Error during prediction pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Liveness probe for Kubernetes / Container Orchestration."""
    return {
        "status": "healthy",
        "model_loaded": "model" in ml_models
    }
