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

# Global state
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_fraud_model.json")
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI."""
    logger.info(f"Loading advanced model from {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        ml_models["model"] = model
        
        logger.info("Initializing SHAP TreeExplainer...")
        ml_models["explainer"] = shap.TreeExplainer(model)
        logger.info("Ready for inference.")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}. Train the model first.")
    
    yield
    ml_models.clear()

app = FastAPI(
    title="Advanced APP Fraud Detection API",
    description="Real-time Financial API serving Extreme Gradient Boosting with SHAP Transparency.",
    version="2.0.0",
    lifespan=lifespan
)

FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.80"))

# Updated Schema matches the new advanced Data Generator features
class TransactionRequest(BaseModel):
    transaction_id: str = Field(...)
    sender_account_id: str = Field(...)
    receiver_account_id: str = Field(...)
    amount_gbp: float = Field(..., ge=0)
    is_new_payee: bool = Field(...)
    device_risk_score: float = Field(..., ge=0, le=100)
    # New Velocity/Engineered Features that we assume the feature-store injects before hitting the model
    time_since_last_tx_seconds: float = Field(..., description="Seconds since the sender's last transfer")
    sender_tx_count_24h: int = Field(..., ge=0, description="Number of transfers by the sender in the last 24h")

class FraudResponse(BaseModel):
    transaction_id: str
    fraud_probability_score: float
    block_transaction: bool
    top_reasons: Dict[str, float]

@app.post("/predict", response_model=FraudResponse)
async def predict_fraud(transaction: TransactionRequest):
    model = ml_models.get("model")
    explainer = ml_models.get("explainer")
    
    if not model or not explainer:
        raise HTTPException(status_code=503, detail="Model unavailable.")

    # Must match train.py drop_cols
    feature_names = [
        'amount_gbp',
        'is_new_payee',
        'device_risk_score',
        'time_since_last_tx_seconds',
        'sender_tx_count_24h'
    ]
    
    features_dict = {
        'amount_gbp': [transaction.amount_gbp],
        'is_new_payee': [1 if transaction.is_new_payee else 0],
        'device_risk_score': [transaction.device_risk_score],
        'time_since_last_tx_seconds': [transaction.time_since_last_tx_seconds],
        'sender_tx_count_24h': [transaction.sender_tx_count_24h]
    }
    
    input_df = pd.DataFrame(features_dict, columns=feature_names)
    
    try:
        prob = float(model.predict_proba(input_df)[0, 1])
        
        # Extract SHAP
        shap_values = explainer.shap_values(input_df)
        instance_shap = shap_values[0] if len(shap_values.shape) == 2 else shap_values
        
        reasons = {name: float(val) for name, val in zip(feature_names, instance_shap)}
        top_reasons = {k: v for k, v in sorted(reasons.items(), key=lambda item: item[1], reverse=True) if v > 0}
        
        if not top_reasons:
            top_reasons = {k: v for k, v in sorted(reasons.items(), key=lambda item: abs(item[1]), reverse=True)}

        block = bool(prob >= FRAUD_THRESHOLD)
        
        logger.info(f"Tx {transaction.transaction_id}: score={prob:.3f}, block={block}")
        
        return FraudResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability_score=prob,
            block_transaction=block,
            top_reasons=top_reasons
        )
            
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
