import pandas as pd
import numpy as np
from faker import Faker
from datetime import timedelta
import os
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_advanced_synthetic_data(num_records: int = 60000, fraud_ratio: float = 0.015) -> pd.DataFrame:
    """
    Generate synthetic UK banking transactions injecting advanced APP fraud features.
    Features include velocity metrics and basic network analysis mimicking a mule network.
    """
    logger.info(f"Generating {num_records} synthetic transactions with advanced features...")
    fake = Faker('en_GB')
    np.random.seed(42)
    Faker.seed(42)
    
    # Generate tightly clustered timestamps (for velocity calculations)
    start_date = pd.Timestamp.now() - pd.Timedelta(days=30)
    timestamps = [start_date + pd.Timedelta(minutes=np.random.exponential(scale=30)) for _ in range(num_records)]
    timestamps.sort()
    
    # Establish a pool of accounts to simulate network behavior
    num_accounts = num_records // 10
    account_pool = [str(fake.random_int(min=100000, max=999999)) for _ in range(num_accounts)]
    
    # Define a small pool of known 'fraudsters' (mule accounts)
    num_mules = max(1, int(num_accounts * 0.05))
    mule_accounts = np.random.choice(account_pool, size=num_mules, replace=False)

    data = {
        'transaction_id': [str(uuid.uuid4()) for _ in range(num_records)],
        'sender_account_id': np.random.choice(account_pool, size=num_records),
        'receiver_account_id': np.random.choice(account_pool, size=num_records),
        'amount_gbp': np.round(np.random.lognormal(mean=4.5, sigma=1.2, size=num_records), 2),
        'timestamp': timestamps,
        'is_new_payee': np.random.choice([0, 1], size=num_records, p=[0.7, 0.3]),
        'device_risk_score': np.round(np.random.uniform(0, 50, size=num_records), 2),
        'is_fraud': np.zeros(num_records, dtype=int)
    }
    
    df = pd.DataFrame(data)
    
    # Prevent self-transfers
    df.loc[df['sender_account_id'] == df['receiver_account_id'], 'receiver_account_id'] = \
        df['receiver_account_id'].apply(lambda _: str(fake.random_int(min=100000, max=999999)))

    # --- ADVANCED FEATURE INJECTION ---
    
    # 1. Network Feature: Multiple senders to a single mule (The "Drop Account" pattern)
    logger.info("Injecting Network Features (Mule accounts)...")
    mule_mask = df['receiver_account_id'].isin(mule_accounts) & (df['is_new_payee'] == 1)
    
    df.loc[mule_mask, 'amount_gbp'] = df.loc[mule_mask, 'amount_gbp'] * 1.5 # mules get larger drops
    df.loc[mule_mask, 'device_risk_score'] = np.clip(df.loc[mule_mask, 'device_risk_score'] + 40, 0, 100)
    
    # 2. Velocity Feature generation (Simulating real-time streaming counters)
    logger.info("Computing Velocity Features...")
    df = df.sort_values(by=['sender_account_id', 'timestamp'])
    
    # Time since last transaction by this sender
    df['time_since_last_tx_seconds'] = df.groupby('sender_account_id')['timestamp'].diff().dt.total_seconds().fillna(86400)
    
    # Count of transactions by this sender in the last 24 hours (simulated using rolling window)
    df = df.set_index('timestamp')
    df['sender_tx_count_24h'] = df.groupby('sender_account_id')['transaction_id'].rolling('24h').count().reset_index(level=0, drop=True)
    df = df.reset_index()

    # Define Velocity Fraud Pattern: Rapid consecutive transfers (e.g. less than 5 minutes apart) to a new payee
    velocity_fraud_mask = (df['time_since_last_tx_seconds'] < 300) & (df['is_new_payee'] == 1) & (df['amount_gbp'] > 1000)
    
    # Finalize Fraud Labels Based on Patterns
    num_frauds_needed = int(num_records * fraud_ratio)
    
    fraud_candidates = df[mule_mask | velocity_fraud_mask].index.tolist()
    
    if len(fraud_candidates) > num_frauds_needed:
        selected_frauds = np.random.choice(fraud_candidates, size=num_frauds_needed, replace=False)
    else:
        selected_frauds = fraud_candidates
        
    df.loc[selected_frauds, 'is_fraud'] = 1
    
    logger.info(f"Target distribution after injection:\n{df['is_fraud'].value_counts(normalize=True)}")
    return df

def main():
    try:
        df = generate_advanced_synthetic_data(num_records=75000, fraud_ratio=0.012)
        
        # Ensure output directory exists
        output_dir = 'data'
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'raw_transactions.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Advanced synthetic data saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error generating advanced data: {str(e)}")
        raise

if __name__ == "__main__":
    main()
