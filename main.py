from fastapi import FastAPI
import joblib
import uvicorn
import numpy as np
import time
import psutil
from pydantic import BaseModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load trained model and preprocessors
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request model
class SpamRequest(BaseModel):
    text: str

# Function to get memory usage 
def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_in_mb = memory_info.rss / 1024 / 1024  #
    return round(memory_usage_in_mb, 2)

@app.get("/")
def home():
    return {"message": "Welcome to the Spam Detection Model"}

@app.post("/predict")
def predict_spam(request: SpamRequest):
    # Start measuring time and memory usage
    start_time = time.time()
    memory_before = get_memory_usage()

    # Transform input text
    text_tfidf = vectorizer.transform([request.text])
    text_scaled = scaler.transform(text_tfidf.toarray())

    # Predict
    prediction = model.predict(text_scaled)[0]
    probability = model.predict_proba(text_scaled)[0][1]

    # End measuring time and memory usage
    end_time = time.time()
    memory_after = get_memory_usage()

    # Calculate response time and memory used
    response_time = round(end_time - start_time, 4)
    memory_used = round(memory_after - memory_before, 2)

    # Log the performance
    logging.info(f"Prediction took {response_time} seconds")
    logging.info(f"Memory before: {memory_before} MB, Memory after: {memory_after} MB, Memory used: {memory_used} MB")

    # Return result with performance metrics
    return {
        "text": request.text,
        "prediction": "spam" if prediction == 1 else "ham",
        "spam_probability": round(float(probability), 4),
        "response_time_seconds": response_time,
        "memory_usage_mb": memory_used
    }

# Run the app (only when executing directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


