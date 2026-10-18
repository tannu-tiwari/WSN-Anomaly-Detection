import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("ml/random_forest_model.pkl")
scaler = joblib.load("ml/scaler.pkl")

print("\nWSN Anomaly Detection")
print("----------------------")

# Take network metrics as input
node_distance = float(input("Node Distance (m): "))
packet_size = int(input("Packet Size (bytes): "))
interval = float(input("Packet Interval (seconds): "))
packets_sent = int(input("Packets Sent: "))
packets_received = int(input("Packets Received: "))
packets_lost = int(input("Packets Lost: "))
pdr = float(input("PDR (%): "))
throughput_bps = float(input("Throughput (bps): "))
average_delay_seconds = float(
    input("Average Delay (seconds): ")
)

# Create input dataframe
input_data = pd.DataFrame([{
    "node_distance": node_distance,
    "packet_size": packet_size,
    "interval": interval,
    "packets_sent": packets_sent,
    "packets_received": packets_received,
    "packets_lost": packets_lost,
    "pdr": pdr,
    "throughput_bps": throughput_bps,
    "average_delay_seconds": average_delay_seconds
}])

# Apply the same scaling used during training
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)[0]

# Prediction probabilities
probabilities = model.predict_proba(input_scaled)[0]

normal_probability = probabilities[0] * 100
anomaly_probability = probabilities[1] * 100

# Display result
if prediction == 1:
    print("\nPrediction: ANOMALY")
else:
    print("\nPrediction: NORMAL")

print(f"Normal Probability: {normal_probability:.2f}%")
print(f"Anomaly Probability: {anomaly_probability:.2f}%")
