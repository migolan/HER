from collections import defaultdict
from copy import deepcopy
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import json

import sys
import os

# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bfp_dqn import BFP_DQN
from src.bfp_env import BFP_ENV, BFPRewardMethod
from src.her import ExperienceReplay, collect_by_policy

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def decaying_epsilon(epsilon_high: float, epsilon_low: float, epsilon_decay: float, epoch: int) -> float:
    return epsilon_low + (epsilon_high - epsilon_low) * np.exp(-epoch / epsilon_decay)

def epsilon_greedy_policy(epsilon_model, q_network: torch.nn.Module, state, epoch: int) -> int:
    with torch.no_grad():
        Q = q_network([state])
    if np.random.rand() < epsilon_model(epoch=epoch):
        return np.random.randint(0, Q.shape[1])
    else:
        return Q.argmax().item()

def train_epoch(sampled_transitions, optimizer: torch.optim.Optimizer, q_network: torch.nn.Module, q_target_network: torch.nn.Module, gamma: float) -> float:
    optimizer.zero_grad()

    states = [t.state for t in sampled_transitions]
    actions = [t.action for t in sampled_transitions]
    pred_q = q_network(states)
    pred_q = pred_q[torch.arange(pred_q.size(0)), actions]

    rewards = torch.Tensor([t.reward for t in sampled_transitions])
    dones = torch.Tensor([BFP_ENV._done(t.next_state) for t in sampled_transitions])
    next_states = [t.next_state for t in sampled_transitions]
    max_q = q_target_network(next_states).max(dim=1)[0]
    target_q = rewards + (1 - dones) * gamma * max_q
    target_q = target_q.detach()

    L = F.smooth_l1_loss(pred_q, target_q)
    L.backward()
    optimizer.step()
    loss = L.detach().item()
    return loss

def training_loop(
        experience_replay: ExperienceReplay,
        optimizer: torch.optim.Optimizer,
        q_network: torch.nn.Module,
        evaluator: callable,
        epochs: int,
        batch_size: int,
        p_her: float,
        q_target_update_rate: int,
        gamma: float
):
    epoch_metrics = defaultdict(list)
    for epoch in tqdm(range(epochs)):
        experience_replay.collect_by_behavior_policy(epoch)
        sampled_transitions = experience_replay.her_sample(batch_size, p_her)
        if epoch % q_target_update_rate == 0:
            q_target_network = deepcopy(q_network)
        epoch_loss = train_epoch(sampled_transitions, optimizer, q_network, q_target_network, gamma)

        _, eval_metrics = evaluator()
        eval_metrics["loss"] = epoch_loss
        for k, v in eval_metrics.items():
            epoch_metrics[k].append(v)

    return epoch_metrics


def save_output(args, metrics, q_network):
    model_filepath = os.path.join('outputs', args.run_name + '.pt')
    torch.save(q_network.state_dict(), model_filepath)

    output_data = {
        "args": vars(args),
        "metrics": metrics,
    }
    results_filepath = os.path.join('outputs', args.run_name + '.json')
    with open(results_filepath, "w") as f:
        json.dump(output_data, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BFP DQN")
    parser.add_argument("--N", type=int, required=True, help="Environment size")
    parser.add_argument("--reward-method", type=str, default="BINARY_REWARD", choices=["BINARY_REWARD", "SHAPED_REWARD"], help="Reward method")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension of DQN")
    parser.add_argument("--epsilon-high", type=float, default=0.9, help="Initial epsilon")
    parser.add_argument("--epsilon-low", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=200, help="Epsilon decay rate")
    parser.add_argument("--episodes-per-epoch", type=int, default=1, help="Episodes per epoch")
    parser.add_argument("--episode-length-factor", type=int, default=1, help="Factor to multiply N to get episode length")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--q-target-update-rate", type=int, default=1000, help="Target network update rate")
    parser.add_argument("--p-her", type=float, required=True, help="HER probability")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--run-name", type=str, required=True, help="Run name")
    return parser.parse_args()


def run_training(args):
    # env
    reward_method = getattr(BFPRewardMethod, args.reward_method)
    env = BFP_ENV(args.N, reward_method, args.episode_length_factor)
    
    # model
    q_network = BFP_DQN(args.N, args.hidden_dim)

    # behavior collector
    epsilon_model = partial(decaying_epsilon, epsilon_high=args.epsilon_high, epsilon_low=args.epsilon_low, epsilon_decay=args.epsilon_decay)
    behavior_policy = partial(epsilon_greedy_policy, q_network=q_network, epsilon_model=epsilon_model)
    experience_replay = ExperienceReplay(env, behavior_policy, args.episodes_per_epoch)

    # optimizer
    optimizer = torch.optim.Adam(q_network.parameters(), lr=args.lr)

    # evaluator
    policy = partial(epsilon_greedy_policy, q_network=q_network, epsilon_model=lambda epoch: 0, epoch=0)
    evaluator = partial(collect_by_policy, env, policy, episodes_per_epoch=5)

    metrics = training_loop(
        experience_replay,
        optimizer,
        q_network,
        evaluator,
        args.epochs,
        args.batch_size,
        args.p_her,
        args.q_target_update_rate,
        args.gamma
    )

    save_output(args, metrics, q_network)


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
