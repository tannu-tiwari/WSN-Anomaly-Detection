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
    page_title="WSN Sentinel | Network Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- VISUAL DESIGN --------------------
st.markdown(
    """
    
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        .stApp {
            background: var(--background-color);
            color: var(--text-color);
        }
        [data-testid="stHeader"] {
            background: var(--background-color);
        }
        [data-testid="stSidebar"] {
            background: var(--secondary-background-color);
            border-right: 1px solid var(--border-color);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--text-color) !important;
        }
        [data-testid="stMetric"] {
            background: var(--secondary-background-color);
            border: 1px solid var(--border-color);
            padding: 15px 17px;
            border-radius: 14px;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: var(--text-color) !important;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
        }
        .hero {
            background: var(--secondary-background-color);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 25px 27px;
            margin-bottom: 20px;
        }
        .eyebrow, .kicker {
            color: #159e8b;
            font-weight: 700;
            letter-spacing: .14em;
            text-transform: uppercase;
        }
        .eyebrow { font-size: .75rem; }
        .kicker { font-size: .72rem; margin-bottom: 3px; }
        .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.15;
            color: var(--text-color);
            margin: 7px 0;
        }
        .hero-sub, .note {
            color: var(--text-color);
            opacity: .75;
        }
        .hero-sub { font-size: .98rem; }
        .note { font-size: .85rem; }
        .pill {
            display: inline-block;
            border: 1px solid var(--border-color);
            background: var(--background-color);
            color: var(--text-color);
            padding: 5px 10px;
            border-radius: 99px;
            font-size: .76rem;
            margin: 12px 6px 0 0;
        }
        .panel-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.08rem;
            font-weight: 600;
            color: var(--text-color);
            margin: 8px 0;
        }
        .good {
            background: var(--secondary-background-color);
            border: 1px solid #278675;
            color: var(--text-color);
            border-radius: 12px;
            padding: 14px 16px;
            font-weight: 600;
        }
        .bad {
            background: var(--secondary-background-color);
            border: 1px solid #b84a58;
            color: var(--text-color);
            border-radius: 12px;
            padding: 14px 16px;
            font-weight: 600;
        }
        .topology {
            background: var(--secondary-background-color);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 12px;
        }
        .stButton > button {
            border-radius: 10px;
            border: 1px solid #278675;
            font-weight: 600;
        }
        hr { border-color: var(--border-color) !important; }
        </style>

            """,
            unsafe_allow_html=True,
        )

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("## 📡 WSN Sentinel")
st.sidebar.caption("Simulation monitoring & ML analytics")
st.sidebar.divider()

if not METRICS_FILE.exists():
    st.error(f"Metrics CSV not found: {METRICS_FILE}")
    st.info("Run the NS-3 simulation first so that wsn_metrics.csv is generated.")
    st.stop()

try:
    df = pd.read_csv(METRICS_FILE)
except Exception as exc:
    st.error(f"Could not read metrics CSV: {exc}")
    st.stop()

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

df = df.dropna(subset=FEATURES).reset_index(drop=True)
if df.empty:
    st.warning("No complete metric rows are available.")
    st.stop()

latest = df.iloc[-1]
latest_run_id = latest["run_id"]
latest_scenario = latest["scenario"]

st.sidebar.write(f"Metric rows available: **{len(df)}**")
if st.sidebar.button("↻ Refresh data", use_container_width=True):
    st.rerun()
st.sidebar.divider()
st.sidebar.caption("Source: NS-3 FlowMonitor CSV")
st.sidebar.caption("Recorded metrics are shown; no live packet stream is claimed.")

# -------------------- HERO --------------------
st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">AD-HOC WIRELESS SENSOR NETWORK · MONITORING</div>
      <div class="hero-title">Network Intelligence Dashboard</div>
      <div class="hero-sub">Network health, ML-based anomaly screening, and simulation analytics.</div>
      <span class="pill">Latest run: {latest_run_id}</span>
      <span class="pill">Scenario: {str(latest_scenario).upper()}</span>
      <span class="pill">NS-3 + Random Forest</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- NETWORK METRICS --------------------
st.markdown('<div class="kicker">01 / Network health</div>', unsafe_allow_html=True)
st.subheader("Latest simulation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Packet Delivery Ratio", f'{float(latest["pdr"]):.2f}%')
c2.metric("Throughput", f'{float(latest["throughput_bps"]):,.2f} bit/s')
c3.metric("Packets Received", f'{int(latest["packets_received"]):,}')
c4.metric("Average Delay", f'{float(latest["average_delay_seconds"]):.6f} s')

c5, c6, c7, c8 = st.columns(4)
c5.metric("Run ID", str(latest["run_id"]))
c6.metric("Scenario", str(latest["scenario"]).upper())
c7.metric("Packets Sent", f'{int(latest["packets_sent"]):,}')
c8.metric("Packets Lost (FlowMonitor)", f'{int(latest["packets_lost"]):,}')

# -------------------- ML PREDICTION --------------------
st.markdown('<div class="kicker">02 / Intelligent detection</div>', unsafe_allow_html=True)
st.subheader("Machine-learning anomaly screening")

if not MODEL_FILE.exists() or not SCALER_FILE.exists():
    st.warning("Trained model or scaler file is missing.")
else:
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)

        # Select the latest contiguous run because run IDs may be reused.
        start_position = len(df) - 1
        while start_position > 0:
            previous = df.iloc[start_position - 1]
            if previous["run_id"] != latest_run_id or previous["scenario"] != latest_scenario:
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
            X_scaled = scaler.transform(run_df[FEATURES])
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            classes = list(model.classes_)
            result_rows = []

            for i, (_, row) in enumerate(run_df.iterrows()):
                probability_map = dict(zip(classes, probabilities[i]))
                normal_prob = float(probability_map.get(0, 0.0))
                anomaly_prob = float(probability_map.get(1, 0.0))
                prediction = int(predictions[i])
                result_rows.append({
                    "Flow ID": row.get("flow_id", "N/A"),
                    "Source": row.get("source", "N/A"),
                    "Destination": row.get("destination", "N/A"),
                    "PDR (%)": round(float(row["pdr"]), 2),
                    "Prediction": "ANOMALY" if prediction == 1 else "NORMAL",
                    "Normal probability (%)": round(normal_prob * 100, 2),
                    "Anomaly probability (%)": round(anomaly_prob * 100, 2),
                })

            results_df = pd.DataFrame(result_rows)
            anomalies = int((predictions == 1).sum())
            normal_count = len(predictions) - anomalies
            p1, p2, p3 = st.columns(3)
            p1.metric("Flows evaluated", len(predictions))
            p2.metric("Normal predictions", normal_count)
            p3.metric("Anomaly predictions", anomalies)

            if anomalies:
                st.markdown(
                    f'<div class="bad">⚠ Anomaly flagged in {anomalies} of {len(predictions)} evaluated flow(s). Review the simulation metrics below.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="good">✓ All {len(predictions)} evaluated flow(s) were classified as NORMAL by the model.</div>',
                    unsafe_allow_html=True,
                )

            st.dataframe(results_df, width="stretch", hide_index=True)
            st.caption(
                "Probabilities are model estimates for each flow, not overall model accuracy. "
                "Predictions are not guaranteed to match the scenario label."
            )
    except Exception as exc:
        st.error(f"Could not run model prediction: {exc}")

# -------------------- NETWORK TOPOLOGY --------------------
st.markdown('<div class="kicker">03 / Network view</div>', unsafe_allow_html=True)
st.subheader("Illustrative 3-node simulation topology")
st.caption("Conceptual view of the configured three-node network, not a live packet animation.")
st.markdown(
    """
    <div class="topology">
    <svg viewBox="0 0 760 230" width="100%" role="img" aria-label="Illustrative three-node ad-hoc network">
      <line x1="155" y1="112" x2="380" y2="55" stroke="#52798a" stroke-width="3"/>
      <line x1="380" y1="55" x2="605" y2="112" stroke="#52798a" stroke-width="3"/>
      <line x1="155" y1="112" x2="605" y2="112" stroke="#e0a44f" stroke-width="3" stroke-dasharray="7 6"/>
      <circle cx="155" cy="112" r="42" fill="#122e2c" stroke="#49b9a7" stroke-width="2"/>
      <circle cx="380" cy="55" r="42" fill="#142331" stroke="#6e9fb4" stroke-width="2"/>
      <circle cx="605" cy="112" r="42" fill="#33231e" stroke="#e0a44f" stroke-width="2"/>
      <g fill="#edf5fb" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="600">
        <text x="155" y="109">Node 0</text><text x="155" y="128" font-size="11" font-weight="400">Source</text>
        <text x="380" y="52">Node 1</text><text x="380" y="71" font-size="11" font-weight="400">Sensor</text>
        <text x="605" y="109">Node 2</text><text x="605" y="128" font-size="11" font-weight="400">Destination</text>
      </g>
      <text x="380" y="205" fill="#8fa5b8" text-anchor="middle" font-family="sans-serif" font-size="12">Illustrative topology · positions are based on simulation configuration</text>
    </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- HISTORY --------------------
st.markdown('<div class="kicker">04 / Simulation history</div>', unsafe_allow_html=True)
st.subheader("Recent simulation records")
history_columns = [
    col for col in [
        "run_id", "scenario", "node_distance", "packet_size", "interval",
        "packets_sent", "packets_received", "packets_lost", "pdr",
        "throughput_bps", "average_delay_seconds",
    ] if col in df.columns
]
st.dataframe(df[history_columns].tail(100).iloc[::-1], width="stretch", hide_index=True)

chart_df = df.copy()
chart_df["run_label"] = chart_df["run_id"].astype(str) + " · " + chart_df.index.astype(str)
left, right = st.columns(2)
with left:
    st.markdown('<div class="panel-title">Packet delivery ratio</div>', unsafe_allow_html=True)
    st.line_chart(chart_df.set_index("run_label")[["pdr"]], y_label="PDR (%)")
with right:
    st.markdown('<div class="panel-title">Throughput</div>', unsafe_allow_html=True)
    st.line_chart(chart_df.set_index("run_label")[["throughput_bps"]], y_label="bit/s")
st.markdown('<div class="panel-title">Average delay</div>', unsafe_allow_html=True)
st.line_chart(chart_df.set_index("run_label")[["average_delay_seconds"]], y_label="seconds")

# -------------------- MODEL EVALUATION --------------------
st.markdown('<div class="kicker">05 / Model evaluation</div>', unsafe_allow_html=True)
st.subheader("Training evaluation artifacts")
if COMPARISON_FILE.exists():
    comparison = pd.read_csv(COMPARISON_FILE)
    st.markdown("**Model comparison**")
    st.dataframe(comparison, width="stretch", hide_index=True)
    model_col = next((c for c in comparison.columns if c.lower() == "model"), None)
    metric_candidates = ["Accuracy", "Precision", "Recall", "F1_Score", "F1 Score"]
    metric_columns = [c for c in metric_candidates if c in comparison.columns]
    if model_col and metric_columns:
        st.bar_chart(comparison.set_index(model_col)[metric_columns])
else:
    st.info("Model comparison file not found.")

if IMPORTANCE_FILE.exists():
    importance = pd.read_csv(IMPORTANCE_FILE)
    st.markdown("**Feature importance**")
    st.dataframe(importance, width="stretch", hide_index=True)
    feature_column = next(
        (col for col in importance.columns if col.lower() in ("feature", "features", "feature_name")),
        None,
    )
    numeric_columns = importance.select_dtypes(include="number").columns.tolist()
    if feature_column and numeric_columns:
        st.bar_chart(importance.set_index(feature_column)[numeric_columns])

st.divider()
st.caption(
    "Research prototype. The training dataset includes synthetic samples, so evaluation scores are preliminary "
    "and do not establish real-world generalization. The topology is illustrative; this dashboard reads recorded "
    "NS-3 metrics rather than a live packet stream."
)
