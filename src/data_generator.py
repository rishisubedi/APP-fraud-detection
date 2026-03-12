import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(num_records: int = 100000, fraud_ratio: float = 0.01) -> pd.DataFrame:
    """
    Generate synthetic UK banking transactions with injected APP fraud patterns.
    
    Args:
        num_records (int): Number of total transactions to generate.
        fraud_ratio (float): Approximate ratio of fraudulent transactions.
        
    Returns:
        pd.DataFrame: DataFrame containing synthetic transactions.
    """
    logger.info(f"Generating {num_records} synthetic transactions...")
    fake = Faker('en_GB')
    np.random.seed(42)
    Faker.seed(42)
    
    # Generate base features
    timestamps = [fake.date_time_between(start_date='-60d', end_date='now') for _ in range(num_records)]
    
    data = {
        'transaction_id': [fake.uuid4() for _ in range(num_records)],
        'account_id': np.random.randint(10000, 99999, size=num_records).astype(str),
        'receiver_account_id': np.random.randint(10000, 99999, size=num_records).astype(str),
        'amount_gbp': np.round(np.random.lognormal(mean=4.0, sigma=1.2, size=num_records), 2),
        'timestamp': timestamps,
        'is_new_payee': np.random.choice([0, 1], size=num_records, p=[0.8, 0.2]),
        'device_risk_score': np.round(np.random.uniform(0, 100, size=num_records), 2),
        'is_fraud': np.zeros(num_records, dtype=int)
    }
    
    df = pd.DataFrame(data)
    
    # Inject APP Fraud Patterns
    # Pattern 1: High value transfers to new payees late at night (11 PM - 4 AM)
    df['hour'] = df['timestamp'].dt.hour
    late_night_mask = (df['hour'] >= 23) | (df['hour'] <= 4)
    high_value_mask = df['amount_gbp'] > 2000
    new_payee_mask = df['is_new_payee'] == 1
    
    pattern1_mask = late_night_mask & high_value_mask & new_payee_mask
    
    # Pattern 2: Very high device risk score with new payee
    pattern2_mask = (df['device_risk_score'] > 90) & (df['is_new_payee'] == 1) & (df['amount_gbp'] > 500)
    
    # Apply fraud labels based on patterns
    num_frauds_needed = int(num_records * fraud_ratio)
    
    fraud_candidates = df[pattern1_mask | pattern2_mask].index
    if len(fraud_candidates) > num_frauds_needed:
        selected_frauds = np.random.choice(fraud_candidates, size=num_frauds_needed, replace=False)
    else:
        selected_frauds = fraud_candidates
        
    df.loc[selected_frauds, 'is_fraud'] = 1
    
    # Also add some random frauds to make it realistic
    remaining_frauds = max(0, num_frauds_needed - len(selected_frauds))
    if remaining_frauds > 0:
        non_fraud_indices = df[df['is_fraud'] == 0].index
        random_frauds = np.random.choice(non_fraud_indices, size=remaining_frauds, replace=False)
        df.loc[random_frauds, 'is_fraud'] = 1

    df = df.drop(columns=['hour'])
    
    logger.info(f"Generated data with target distribution:\n{df['is_fraud'].value_counts(normalize=True)}")
    return df

def main():
    try:
        df = generate_synthetic_data(num_records=50000, fraud_ratio=0.015)
        
        # Ensure output directory exists
        output_dir = 'data'
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'transactions.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Synthetic data saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error generating data: {str(e)}")
        raise

if __name__ == "__main__":
    main()
