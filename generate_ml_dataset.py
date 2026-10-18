import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. LOAD NS-3 GENERATED DATA
# --------------------------------------------------

INPUT_FILE = "ns-3-dev/data/wsn_metrics.csv"
OUTPUT_FILE = "ns-3-dev/data/wsn_ml_dataset.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("Original NS-3 records:", len(df))
print("=" * 60)


# --------------------------------------------------
# 2. IDENTIFY ORIGINAL ANOMALIES
# --------------------------------------------------

# A record with zero received packets / zero PDR
# represents a communication failure.
df["label"] = 0

df.loc[
    (df["packets_received"] == 0) |
    (df["pdr"] == 0),
    "label"
] = 1


# --------------------------------------------------
# 3. GENERATE NORMAL SYNTHETIC DATA
# --------------------------------------------------

np.random.seed(42)

normal_source = df[df["label"] == 0].copy()

normal_records = []

TARGET_NORMAL = 140

for i in range(TARGET_NORMAL):

    row = normal_source.sample(1).iloc[0].copy()

    # Small realistic variation
    row["node_distance"] = round(
        max(30, row["node_distance"] + np.random.normal(0, 2)), 2
    )

    row["packet_size"] = int(
        row["packet_size"]
    )

    row["interval"] = round(
        max(0.1, row["interval"] + np.random.normal(0, 0.03)), 3
    )

    row["packets_sent"] = int(row["packets_sent"])

    # Normal network usually has very small packet loss
    loss_probability = np.random.choice(
        [0, 0.01, 0.02],
        p=[0.85, 0.10, 0.05]
    )

    lost = int(
        round(row["packets_sent"] * loss_probability)
    )

    row["packets_lost"] = lost
    row["packets_received"] = row["packets_sent"] - lost

    if row["packets_sent"] > 0:
        row["pdr"] = round(
            (row["packets_received"] / row["packets_sent"]) * 100,
            2
        )
    else:
        row["pdr"] = 0

    # Small variation in throughput and delay
    row["throughput_bps"] = round(
        max(
            0,
            row["throughput_bps"] *
            np.random.uniform(0.90, 1.10)
        ),
        2
    )

    row["average_delay_seconds"] = round(
        max(
            0,
            row["average_delay_seconds"] *
            np.random.uniform(0.90, 1.15)
        ),
        8
    )

    row["run_id"] = 1000 + i
    row["scenario"] = "synthetic_normal"
    row["flow_id"] = int(row["flow_id"])
    row["label"] = 0

    normal_records.append(row)


normal_df = pd.DataFrame(normal_records)


# --------------------------------------------------
# 4. GENERATE ANOMALY DATA
# --------------------------------------------------

anomaly_records = []

TARGET_ANOMALIES = 60

for i in range(TARGET_ANOMALIES):

    row = normal_source.sample(1).iloc[0].copy()

    row["run_id"] = 2000 + i
    row["scenario"] = "synthetic_anomaly"

    anomaly_type = np.random.choice(
        [
            "packet_loss",
            "high_delay",
            "low_throughput",
            "communication_failure"
        ]
    )

    # ----------------------------------------------
    # Packet Loss Anomaly
    # ----------------------------------------------

    if anomaly_type == "packet_loss":

        row["packets_lost"] = int(
            row["packets_sent"] *
            np.random.uniform(0.30, 0.70)
        )

        row["packets_received"] = (
            row["packets_sent"] -
            row["packets_lost"]
        )

        row["pdr"] = round(
            (row["packets_received"] /
             row["packets_sent"]) * 100,
            2
        )

        row["throughput_bps"] = round(
            row["throughput_bps"] *
            np.random.uniform(0.30, 0.70),
            2
        )

        row["average_delay_seconds"] = round(
            row["average_delay_seconds"] *
            np.random.uniform(1.5, 3.0),
            8
        )


    # ----------------------------------------------
    # High Delay Anomaly
    # ----------------------------------------------

    elif anomaly_type == "high_delay":

        row["average_delay_seconds"] = round(
            row["average_delay_seconds"] *
            np.random.uniform(3.0, 8.0),
            8
        )

        row["throughput_bps"] = round(
            row["throughput_bps"] *
            np.random.uniform(0.60, 0.90),
            2
        )

        row["pdr"] = round(
            np.random.uniform(70, 95),
            2
        )


    # ----------------------------------------------
    # Low Throughput Anomaly
    # ----------------------------------------------

    elif anomaly_type == "low_throughput":

        row["throughput_bps"] = round(
            row["throughput_bps"] *
            np.random.uniform(0.20, 0.50),
            2
        )

        row["average_delay_seconds"] = round(
            row["average_delay_seconds"] *
            np.random.uniform(1.5, 3.5),
            8
        )

        row["pdr"] = round(
            np.random.uniform(75, 95),
            2
        )


    # ----------------------------------------------
    # Communication Failure
    # ----------------------------------------------

    else:

        row["packets_received"] = 0
        row["packets_lost"] = row["packets_sent"]
        row["pdr"] = 0
        row["throughput_bps"] = 0
        row["average_delay_seconds"] = 0


    row["flow_id"] = int(row["flow_id"])
    row["label"] = 1

    anomaly_records.append(row)


anomaly_df = pd.DataFrame(anomaly_records)


# --------------------------------------------------
# 5. COMBINE DATA
# --------------------------------------------------

original_df = df.copy()

final_df = pd.concat(
    [
        original_df,
        normal_df,
        anomaly_df
    ],
    ignore_index=True
)


# --------------------------------------------------
# 6. SHUFFLE DATASET
# --------------------------------------------------

final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# 7. SAVE FINAL DATASET
# --------------------------------------------------

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 8. DISPLAY SUMMARY
# --------------------------------------------------

print("=" * 60)
print("Dataset generation completed.")
print("Original NS-3 records :", len(original_df))
print("Synthetic normal     :", len(normal_df))
print("Synthetic anomalies  :", len(anomaly_df))
print("Final records        :", len(final_df))
print("=" * 60)

print("\nLabel distribution:")
print(final_df["label"].value_counts())

print("\nScenario distribution:")
print(final_df["scenario"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)
