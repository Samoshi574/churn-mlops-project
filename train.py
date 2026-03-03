import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Initialize DAGsHub MLflow connection
dagshub.init(repo_owner="Samoshi574", repo_name="churn-mlops-project", mlflow=True)

# Load dataset
data = pd.read_csv("data/churn_data.csv")

# Simple preprocessing (example)
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

# MLflow logging
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

print("Training complete. Accuracy:", accuracy)