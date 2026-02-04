import json
import argparse
from matplotlib import pyplot as plt
import os
import sys

# Add the project root to sys.path to allow importing from src if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def smooth(values, window=50):
    if len(values) < window:
        return values
    
    smoothed = []
    for i in range(len(values) - window + 1):
        chunk = values[i : i + window]
        smoothed.append(sum(chunk) / window)
    return smoothed

def plot_experiment(experiment_name=None, run_names=None):
    if not run_names and experiment_name:
        print(f"No run names provided. Loading from experiments/{experiment_name}.json...")
        experiment_filepath = os.path.join("experiments", experiment_name + ".json")
        try:
            with open(experiment_filepath, 'r') as f:
                configs = json.load(f)
            run_names = [c.get("--run-name") or c.get("run-name") for c in configs]
            run_names = [n for n in run_names if n is not None]
        except Exception as e:
            print(f"Error loading experiment file: {e}")
            sys.exit(1)

    if not run_names:
        print("Error: No run names available to plot.")
        sys.exit(1)

    data_list = []
    for run_name in run_names:
        filepath = os.path.join("outputs", run_name + ".json")
        try:
            data = load_data(filepath)
            data_list.append({"name": run_name, "data": data})
        except Exception as e:
            print(f"Warning: Could not load data for run {run_name}: {e}")

    if not data_list:
        print("Error: No data loaded for any run.")
        sys.exit(1)

    # Assume all files have same metrics keys
    keys = data_list[0]["data"]["metrics"].keys()
    
    n = len(keys)
    # 2 columns
    cols = 2
    rows = (n + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, key in enumerate(keys):
        ax = axes[idx]
        for item in data_list:
            values = item["data"]["metrics"][key]
            smoothed = smooth(values)
            # Use filename params for legend
            label = os.path.basename(item["name"])
            if label.startswith("results_"):
                label = label[8:]
            if label.endswith(".json"):
                label = label[:-5]
            # Make it more readable
            label = label.replace("_--", ", --")
            
            ax.plot(smoothed, label=label)
        
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(True)
    
    # Hide unused axes
    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if experiment_name:
        experiment_plot_filepath = os.path.join("outputs", experiment_name + ".png")
        plt.savefig(experiment_plot_filepath)
        print(f"Plot saved to {experiment_plot_filepath}")
    plt.show()

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-names", nargs="+", help="Experiment run names")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name")
    args = parser.parse_args()
    return args.experiment_name, args.run_names

if __name__ == "__main__":
    experiment_name, run_names = _parse_args()
    plot_experiment(experiment_name, run_names)
