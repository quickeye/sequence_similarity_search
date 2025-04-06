import json
import matplotlib.pyplot as plt
import os

def visualize_flow(flow, prefix):
    x = list(range(len(flow)))
    y = [step["layer"] for step in flow]
    labels = [step["step_type"] for step in flow]

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker='o', linestyle='-', label=prefix)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (x[i], y[i] + 0.1), fontsize=8, rotation=30)
    plt.title(f"Flow Structure: {prefix}")
    plt.xlabel("Step Index")
    plt.ylabel("Layer")
    plt.ylim(-1, max(y) + 2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    path = "../data/example_flows.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run generate_sample_flows.py first.")

    with open(path, "r") as f:
        flows = json.load(f)

    for prefix, flow in flows.items():
        print(f"🔍 Visualizing flow for {prefix}")
        visualize_flow(flow, prefix)
