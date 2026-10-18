
from pathlib import Path

import joblib
import pandas as pd

# Resolve paths relative to this script, not the terminal directory.
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "ns-3-dev" / "data" / "wsn_metrics.csv"
MODEL_FILE = BASE_DIR / "ml" / "random_forest_model.pkl"
SCALER_FILE = BASE_DIR / "ml" / "scaler.pkl"

FEATURES = [
    "node_distance",
    "packet_size",
    "interval",
    "packets_sent",
    "packets_received",
    "packets_lost",
    "pdr",
    "throughput_bps",
    "average_delay_seconds",
]

REQUIRED_COLUMNS = ["run_id", "scenario", *FEATURES]


def main():
    for path in (CSV_FILE, MODEL_FILE, SCALER_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(CSV_FILE)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    if df.empty:
        raise ValueError("The NS-3 metrics CSV is empty.")

    # Find the most recent contiguous run at the end of the CSV.
    # This avoids accidentally combining older runs that reused a run ID.
    last = df.iloc[-1]
    run_id = last["run_id"]
    scenario = last["scenario"]

    same_run = (
        (df["run_id"] == run_id)
        & (df["scenario"] == scenario)
    )
    last_position = len(df) - 1
    start_position = last_position

    while start_position > 0:
        previous = df.iloc[start_position - 1]
        if (
            previous["run_id"] != run_id
            or previous["scenario"] != scenario
        ):
            break
        start_position -= 1

    run_df = df.iloc[start_position:last_position + 1].copy()
    run_df = run_df.loc[same_run.iloc[start_position:last_position + 1].values]
    run_df = run_df.dropna(subset=FEATURES)

    if run_df.empty:
        raise ValueError("Latest run has no complete metric rows.")

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    # Preserve feature order used during training.
    X = run_df[FEATURES]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    classes = list(model.classes_)

    print("\nWSN → ML Anomaly Detection")
    print("=" * 48)
    print(f"Run ID: {run_id}")
    print(f"Scenario: {scenario}")
    print(f"Flows evaluated: {len(run_df)}")

    for index, (_, row) in enumerate(run_df.iterrows()):
        probability_map = dict(zip(classes, probabilities[index]))
        normal_prob = float(probability_map.get(0, 0.0)) * 100
        anomaly_prob = float(probability_map.get(1, 0.0)) * 100
        prediction = int(predictions[index])

        flow_id = row.get("flow_id", "N/A")
        source = row.get("source", "N/A")
        destination = row.get("destination", "N/A")

        status = "ANOMALY" if prediction == 1 else "NORMAL"

        print("\n" + "-" * 48)
        print(f"Flow ID: {flow_id}")
        print(f"Source → Destination: {source} → {destination}")
        print(f"Packets sent: {int(row['packets_sent'])}")
        print(f"Packets received: {int(row['packets_received'])}")
        print(f"PDR: {row['pdr']:.2f}%")
        print(f"Throughput: {row['throughput_bps']:.2f} bit/s")
        print(f"Average delay: {row['average_delay_seconds']:.6f} s")
        print(f"Prediction: {status}")
        print(f"Normal probability: {normal_prob:.2f}%")
        print(f"Anomaly probability: {anomaly_prob:.2f}%")

    print("\nNote: probabilities are model estimates, not accuracy scores.")


if __name__ == "__main__":
    main()
