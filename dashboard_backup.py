
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -------------------- PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent
METRICS_FILE = BASE_DIR / "ns-3-dev" / "data" / "wsn_metrics.csv"
MODEL_FILE = BASE_DIR / "ml" / "random_forest_model.pkl"
SCALER_FILE = BASE_DIR / "ml" / "scaler.pkl"
COMPARISON_FILE = BASE_DIR / "ml" / "model_comparison.csv"
IMPORTANCE_FILE = BASE_DIR / "ml" / "feature_importance.csv"

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

st.set_page_config(
    page_title="WSN Anomaly Detection",
    page_icon="📡",
    layout="wide",
)

st.title("📡 WSN Network Anomaly Detection")
st.caption("NS-3 Simulation • Random Forest • Machine Learning")

# -------------------- LOAD DATA --------------------
if not METRICS_FILE.exists():
    st.error(f"Metrics CSV not found: {METRICS_FILE}")
    st.stop()

df = pd.read_csv(METRICS_FILE)

required = [
    "run_id", "scenario", "packets_sent", "packets_received",
    "packets_lost", "pdr", "throughput_bps",
    "average_delay_seconds", *FEATURES,
]

missing = [column for column in required if column not in df.columns]
if missing:
    st.error(f"Required columns missing from metrics CSV: {missing}")
    st.stop()

if df.empty:
    st.warning("No simulation results available yet.")
    st.stop()

# Latest row is displayed; all records remain available in history.
df = df.dropna(subset=FEATURES).reset_index(drop=True)
if df.empty:
    st.warning("No complete metric rows available.")
    st.stop()

latest = df.iloc[-1]

st.sidebar.header("Dashboard Controls")
st.sidebar.write(f"Available metric rows: {len(df)}")
if st.sidebar.button("Refresh data"):
    st.rerun()

st.subheader("Latest Simulation")
c1, c2, c3 = st.columns(3)
c1.metric("Run ID", str(latest["run_id"]))
c2.metric("Scenario", str(latest["scenario"]).upper())
c3.metric("PDR", f'{latest["pdr"]:.2f}%')

c4, c5, c6 = st.columns(3)
c4.metric("Throughput", f'{latest["throughput_bps"]:.2f} bit/s')
c5.metric("Packets Received", int(latest["packets_received"]))
c6.metric("Average Delay", f'{latest["average_delay_seconds"]:.6f} s')


# -------------------- ML PREDICTION --------------------
st.subheader("Machine Learning Prediction")

if not MODEL_FILE.exists() or not SCALER_FILE.exists():
    st.warning("Trained model or scaler file is missing.")
else:
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)

        # Identify the latest contiguous simulation run
        latest_row = df.iloc[-1]
        latest_run_id = latest_row["run_id"]
        latest_scenario = latest_row["scenario"]

        start_position = len(df) - 1

        while start_position > 0:
            previous = df.iloc[start_position - 1]

            if (
                previous["run_id"] != latest_run_id
                or previous["scenario"] != latest_scenario
            ):
                break

            start_position -= 1

        run_df = df.iloc[start_position:].copy()

        run_df = run_df[
            (run_df["run_id"] == latest_run_id)
            & (run_df["scenario"] == latest_scenario)
        ].dropna(subset=FEATURES)

        if run_df.empty:
            st.warning("No complete flows found in the latest run.")
        else:
            X = run_df[FEATURES]
            X_scaled = scaler.transform(X)

            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            classes = list(model.classes_)

            st.write(
                f"Latest Run: **{latest_run_id}** | "
                f"Scenario: **{latest_scenario}** | "
                f"Flows Evaluated: **{len(run_df)}**"
            )

            result_rows = []

            for i, (_, row) in enumerate(run_df.iterrows()):
                probability_map = dict(
                    zip(classes, probabilities[i])
                )

                normal_prob = float(
                    probability_map.get(0, 0.0)
                )
                anomaly_prob = float(
                    probability_map.get(1, 0.0)
                )
                prediction = int(predictions[i])

                result_rows.append({
                    "Flow ID": row.get("flow_id", "N/A"),
                    "Source": row.get("source", "N/A"),
                    "Destination": row.get("destination", "N/A"),
                    "PDR (%)": row["pdr"],
                    "Prediction": (
                        "ANOMALY" if prediction == 1
                        else "NORMAL"
                    ),
                    "Normal Probability (%)": round(
                        normal_prob * 100, 2
                    ),
                    "Anomaly Probability (%)": round(
                        anomaly_prob * 100, 2
                    ),
                })

            results_df = pd.DataFrame(result_rows)

            anomalies = int((predictions == 1).sum())
            normal_count = len(predictions) - anomalies

            m1, m2, m3 = st.columns(3)
            m1.metric("Flows Evaluated", len(predictions))
            m2.metric("Normal Predictions", normal_count)
            m3.metric("Anomaly Predictions", anomalies)

            st.dataframe(
                results_df,
                width="stretch",
                hide_index=True,
            )

            if anomalies:
                st.error(
                    f"⚠️ ML detected anomalies in {anomalies} "
                    f"of {len(predictions)} evaluated flow(s)."
                )
            else:
                st.success(
                    f"All {len(predictions)} evaluated flow(s) "
                    "were classified as NORMAL."
                )

            st.caption(
                "Probabilities are model estimates for each flow, "
                "not overall model accuracy. Predictions are not "
                "guaranteed to match the simulation scenario label."
            )

    except Exception as exc:
        st.error(f"Could not run model prediction: {exc}")


# -------------------- HISTORY --------------------
st.subheader("Simulation History")

history_columns = [
    "run_id", "scenario", "node_distance", "packet_size",
    "interval", "packets_sent", "packets_received",
    "packets_lost", "pdr", "throughput_bps",
    "average_delay_seconds",
]
st.dataframe(
    df[history_columns].tail(100).iloc[::-1],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Network Metrics by Run")
chart_df = df.copy()
chart_df["run_label"] = chart_df["run_id"].astype(str)

st.write("Packet Delivery Ratio (%)")
st.line_chart(chart_df.set_index("run_label")[["pdr"]])

st.write("Throughput (bit/s)")
st.line_chart(chart_df.set_index("run_label")[["throughput_bps"]])

st.write("Average Delay (seconds)")
st.line_chart(chart_df.set_index("run_label")[["average_delay_seconds"]])

# -------------------- MODEL EVALUATION --------------------
st.subheader("Model Evaluation")

if COMPARISON_FILE.exists():
    comparison = pd.read_csv(COMPARISON_FILE)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    metric_columns = [
        column for column in
        ["Accuracy", "Precision", "Recall", "F1_Score"]
        if column in comparison.columns
    ]
    if metric_columns:
        st.bar_chart(comparison.set_index("Model")[metric_columns])

if IMPORTANCE_FILE.exists():
    importance = pd.read_csv(IMPORTANCE_FILE)
    st.write("Feature Importance")
    st.dataframe(importance, use_container_width=True, hide_index=True)

    numeric_columns = importance.select_dtypes(include="number").columns
    if len(numeric_columns):
        feature_column = next(
            (col for col in importance.columns
             if col.lower() in ("feature", "features", "feature_name")),
            None,
        )
        if feature_column:
            st.bar_chart(importance.set_index(feature_column)[numeric_columns])

st.caption(
    "Research prototype: results depend on the simulation data and "
    "training dataset. Synthetic training samples may limit real-world "
    "generalization."
)
