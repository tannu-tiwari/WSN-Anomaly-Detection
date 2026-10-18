import pandas as pd
import matplotlib.pyplot as plt

# Load cross-validation results
df = pd.read_csv("ml/cross_validation_results.csv")

# Metrics to compare
metrics = [
    "Accuracy Mean",
    "Precision Mean",
    "Recall Mean",
    "F1 Mean"
]

# Convert values to percentage
for metric in metrics:
    df[metric] = df[metric] * 100

# Create separate graph for each metric
for metric in metrics:
    plt.figure(figsize=(8, 5))

    plt.bar(df["Model"], df[metric])

    plt.title(f"Model Comparison - {metric.replace(' Mean', '')}")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("Score (%)")
    plt.ylim(0, 105)

    plt.tight_layout()

    # Create filename
    filename = metric.lower().replace(" ", "_")
    filename = f"ml/{filename}.png"

    plt.savefig(filename)
    plt.close()

    print(f"Saved: {filename}")

print("\nAll model comparison graphs generated successfully!")
