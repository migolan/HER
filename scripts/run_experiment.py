import os
import subprocess
import json
import argparse
import sys

from tqdm import tqdm


def subprocess_wrapper(cmd):
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Plot generation failed with exit code {e.returncode}")
        print(f"Command run: {' '.join(cmd)}")
        sys.exit(1)


def run_run(**kwargs):
    cmd = ["python", "scripts/train.py"]
    for k, v in kwargs.items():
        cmd.extend([k, str(v)])
        if k == "--run-name":
            run_name = v

    subprocess_wrapper(cmd)
    return run_name

def main():
    parser = argparse.ArgumentParser(description="Run experiment of multiple runs from an experiment run configs JSON file")
    parser.add_argument("--experiment-name", type=str, default="mini_experiment", help="Experiment name")
    args = parser.parse_args()

    # Load experiment run configs
    experiment_filepath = os.path.join('experiments', args.experiment_name + '.json')
    with open(experiment_filepath, 'r') as f:
        experiment_run_configs = json.load(f)

    # Run experiment runs
    for run_config in tqdm(experiment_run_configs):
        run_run(**run_config)
    print("All experiment runs finished.")

    # Generate runs comparison plots
    cmd_plot = ["python", "scripts/plot_experiment.py", "--experiment-name", args.experiment_name]
    subprocess_wrapper(cmd_plot)

if __name__ == "__main__":
    main()
