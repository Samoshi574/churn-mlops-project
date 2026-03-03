from fastapi import FastAPI
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load trained model
model = joblib.load("models/model.pkl")

# Home route
@app.get("/")
def home():
    return {"message": "Churn Prediction API is Running"}

# Prediction route
@app.post("/predict")
def predict(age: int, salary: float):
    input_data = np.array([[age, salary]])
    prediction = model.predict(input_data)[0]
    return {"churn_prediction": int(prediction)}