# Use the official lightweight Python 3.10 image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files to the working directory
COPY . /app

# Define environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/xgb_fraud_model.json
ENV PYTHONUNBUFFERED=1

# Expose the API port
EXPOSE 8000

# Start the FastAPI application with Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
