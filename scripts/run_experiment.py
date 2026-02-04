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
    cmd = ["python", "scripts/_train.py"]
    for k, v in kwargs.items():
        cmd.extend([k, str(v)])
        if k == "--run-name":
            run_name = v

    subprocess_wrapper(cmd)
    return run_name

def load_experiment_run_configs(experiment_name):
    experiment_filepath = os.path.join('experiments', experiment_name + '.json')
    with open(experiment_filepath, 'r') as f:
        experiment_run_configs = json.load(f)
    return experiment_run_configs

def run_experiment(experiment_name):
    experiment_run_configs = load_experiment_run_configs(experiment_name)

    # Run experiment runs
    for run_config in tqdm(experiment_run_configs):
        run_run(**run_config)
    print("All experiment runs finished.")

    # Generate runs comparison plots
    cmd_plot = ["python", "scripts/_plot_experiment.py", "--experiment-name", experiment_name]
    subprocess_wrapper(cmd_plot)

def _parse_args():
    parser = argparse.ArgumentParser(description="Run experiment of multiple runs from an experiment run configs JSON file")
    parser.add_argument("--experiment-name", type=str, default="mini_experiment", help="Experiment name")
    args = parser.parse_args()
    return args.experiment_name

if __name__ == "__main__":
    experiment_name = _parse_args()
    run_experiment(experiment_name)
