import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Load test dataset
df = pd.read_csv("ml/test.csv")

# Separate features and target
X_test = df.drop("label", axis=1)
y_test = df["label"]

# Load trained model and scaler
model = joblib.load("ml/random_forest_model.pkl")
scaler = joblib.load("ml/scaler.pkl")

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Anomaly"]
))

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"]
)

display.plot()
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

# Save graph
plt.savefig("ml/confusion_matrix.png")

print("\nConfusion matrix saved to: ml/confusion_matrix.png")
