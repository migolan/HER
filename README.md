# HER BFP DQN Experimentation Framework

This project implements Hindsight Experience Replay (HER), demonstrated on the Bit-Flipping Problem (BFP) with Deep Q-Learning (DQN).

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
  - [Running Experiments](#running-experiments)
  - [Training a Single Model](#training-a-single-model)
  - [Comparing Runs](#comparing-runs)
- [Project Structure](#project-structure)

## Setup

Ensure you have the required dependencies:
```bash
pip install torch numpy tqdm matplotlib
```

## Usage

### Running Experiments
The `run_experiment.py` script executes a batch of runs defined in a JSON config file located in the `experiments/` directory.

```bash
python scripts/run_experiment.py --experiment-name experiment_name
```

**Arguments:**
- `--experiment-name`: Name of the experiment JSON file in the `experiments/` folder (without `.json`). Defaults to `mini_experiment`.

### Training a Single Model
You can run individual training sessions using `_train.py`.

```bash
python scripts/_train.py --N 5 --p-her 0.8 --n-epochs 1000 --run-name name
```

**Key Arguments:**
- `--N`: Environment size (number of bits).
- `--p-her`: Probability of using HER samples.
- `--n-epochs`: Number of training epochs.
- `--run-name`: Run name which will be used to save the results.
- Run `python scripts/_train.py --help` for a full list of parameters.

### Comparing Runs
If you have multiple runs files in JSON format, you can compare them manually using `_plot_experiment.py`:

```bash
python scripts/_plot_experiment.py --run-names run_name1 run_name2 --experiment-name experiment_name
```

## Project Structure

```text
her/
├── README.md                # Project description and setup instructions
├── scripts/
│   ├── run_experiment.py    # Orchestrates multiple training runs
│   ├── train.py            # Training script for a single model
│   └── plot_experiment.py  # Utility for comparing and plotting results
├── src/
│   ├── bfp_dqn.py           # DQN model architecture
│   ├── bfp_env.py           # Bit-Flipping Problem environment logic
│   └── her.py               # Hindsight Experience Replay implementation
├── experiments/
│   └── *.json               # Experiment configuration files (e.g., mini_experiment.json)
└── outputs/                 # Storage for results (.json), plots (.png) and models (.pt)
```
