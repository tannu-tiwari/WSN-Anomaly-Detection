import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD TRAINING AND TEST DATA
# ============================================================

train = pd.read_csv("ml/train.csv")
test = pd.read_csv("ml/test.csv")

X_train = train.drop("label", axis=1)
y_train = train["label"]

X_test = test.drop("label", axis=1)
y_test = test["label"]

print("=" * 60)
print("ML DATA LOADED")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 2. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        random_state=42,
        max_iter=1000
    )
}


# ============================================================
# 3. TRAIN AND EVALUATE EACH MODEL
# ============================================================

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # Display metrics
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Classification Report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })


# ============================================================
# 4. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# ============================================================
# 5. SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(
    "ml/model_comparison.csv",
    index=False
)

print("\nResults saved to:")
print("ml/model_comparison.csv")

print("=" * 60)
print("ML TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)
