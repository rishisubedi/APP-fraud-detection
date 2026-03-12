import random
import uuid
from locust import HttpUser, task, between, events

class APPFraudStressTest(HttpUser):
    # Simulate realistic API call gaps (between 0.1 and 0.5 seconds)
    wait_time = between(0.1, 0.5)

    @task
    def predict_fraud(self):
        """
        Generates random, realistic JSON payloads to prevent API caching
        and stress tests the /predict endpoint.
        """
        # Dynamic payload generation matching the API schema
        payload = {
            "transaction_id": str(uuid.uuid4()),
            "account_id": str(random.randint(10000, 99999)),
            "receiver_account_id": str(random.randint(10000, 99999)),
            # Randomize amount between £5 and £5000
            "amount_gbp": round(random.uniform(5.0, 5000.0), 2),
            # 20% chance of being a new payee
            "is_new_payee": random.choices([True, False], weights=[0.2, 0.8])[0],
            # Continuous risk score from 0.0 to 100.0 (API schema specifies 0-100)
            "device_risk_score": round(random.uniform(0.0, 100.0), 2)
        }

        headers = {
            "Content-Type": "application/json"
        }

        # Use the name parameter to group requests under '/predict' in Locust UI
        with self.client.post("/predict", json=payload, headers=headers, name="/predict", catch_response=True) as response:
            try:
                # Expecting a 200 OK response with the correct schema
                if response.status_code == 200:
                    json_response = response.json()
                    
                    # Validate that critical fields exist in the response
                    if "fraud_probability_score" in json_response and "block_transaction" in json_response:
                        response.success()
                    else:
                        response.failure(f"Missing required fields in response: {json_response}")
                else:
                    # Log unexpected HTTP error codes (e.g., 500 Internal Server Error)
                    response.failure(f"HTTP ERROR {response.status_code}: {response.text}")
                    
            except Exception as e:
                # Log JSON parsing errors or connection timeouts
                response.failure(f"Exception during request: {str(e)}")

# Optional: Add an event listener to log when the test starts/stops
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Starting APP Fraud Detection Stress Test...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Stress Test Completed.")
