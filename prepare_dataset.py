import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

INPUT_FILE = "ns-3-dev/data/wsn_ml_dataset.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("Original dataset shape:", df.shape)
print("=" * 60)


# --------------------------------------------------
# 2. SELECT ML FEATURES
# --------------------------------------------------

features = [
    "node_distance",
    "packet_size",
    "interval",
    "packets_sent",
    "packets_received",
    "packets_lost",
    "pdr",
    "throughput_bps",
    "average_delay_seconds"
]

X = df[features]
y = df["label"]


# --------------------------------------------------
# 3. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 5. CONVERT BACK TO DATAFRAMES
# --------------------------------------------------

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=features
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=features
)

X_train_scaled["label"] = y_train.reset_index(drop=True)
X_test_scaled["label"] = y_test.reset_index(drop=True)


# --------------------------------------------------
# 6. SAVE PROCESSED DATA
# --------------------------------------------------

X_train_scaled.to_csv(
    "ml/train.csv",
    index=False
)

X_test_scaled.to_csv(
    "ml/test.csv",
    index=False
)


# --------------------------------------------------
# 7. DISPLAY SUMMARY
# --------------------------------------------------

print("\nSelected features:")
for feature in features:
    print("-", feature)

print("\nTraining data:", X_train_scaled.shape)
print("Testing data :", X_test_scaled.shape)

print("\nTraining labels:")
print(X_train_scaled["label"].value_counts())

print("\nTesting labels:")
print(X_test_scaled["label"].value_counts())

print("\nSaved files:")
print("ml/train.csv")
print("ml/test.csv")

print("=" * 60)
print("Preprocessing completed successfully.")
print("=" * 60)
