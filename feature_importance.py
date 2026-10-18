import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("ml/train.csv")

X_train = train.drop("label", axis=1)
y_train = train["label"]


# ============================================================
# 2. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# 3. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(importance.to_string(index=False))


# ============================================================
# 4. SAVE RESULTS
# ============================================================

importance.to_csv(
    "ml/feature_importance.csv",
    index=False
)


# ============================================================
# 5. CREATE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Network Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "ml/feature_importance.png",
    dpi=300
)

plt.show()


print("\nSaved:")
print("ml/feature_importance.csv")
print("ml/feature_importance.png")
