import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Load training dataset
df = pd.read_csv("ml/train.csv")

# Separate features and target
X = df.drop("label", axis=1)
y = df["label"]

# Define models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"]
    )

    results.append({
        "Model": name,
        "Accuracy Mean": scores["test_accuracy"].mean(),
        "Precision Mean": scores["test_precision"].mean(),
        "Recall Mean": scores["test_recall"].mean(),
        "F1 Mean": scores["test_f1"].mean()
    })

# Create results table
results_df = pd.DataFrame(results)

print("\nCross-Validation Results:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("ml/cross_validation_results.csv", index=False)

print("\nResults saved to: ml/cross_validation_results.csv")
