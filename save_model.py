import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load training data
df = pd.read_csv("ml/train.csv")

# Separate features and target
X = df.drop("label", axis=1)
y = df["label"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train final Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "ml/random_forest_model.pkl")
joblib.dump(scaler, "ml/scaler.pkl")

print("Random Forest model saved successfully!")
print("Model: ml/random_forest_model.pkl")
print("Scaler: ml/scaler.pkl")
