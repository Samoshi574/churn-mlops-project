import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("data/churn_data.csv")

# Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set experiment
mlflow.set_experiment("Churn_Project")

# Start MLflow run
with mlflow.start_run():

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Log accuracy
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", accuracy)

    # Save local model
    joblib.dump(model, "models/model.pkl")