from collections import defaultdict
from copy import deepcopy
from functools import partial
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import json
import simple_parsing

import sys
import os

# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bfp_dqn import BFP_DQN
from src.bfp_env import BFP_ENV, BFPRewardMethod
from src.her import ExperienceReplay, collect_by_policy

@dataclass
class TrainConfig:
    # Required arguments
    N: int
    p_her: float
    run_name: str
    
    # Optional arguments with defaults
    reward_method: str = "BINARY_REWARD"
    hidden_dim: int = 128
    epsilon_high: float = 0.9
    epsilon_low: float = 0.05
    epsilon_decay: float = 200.0
    episodes_per_epoch: int = 1
    episode_length_factor: int = 1
    gamma: float = 0.99
    epochs: int = 5000
    lr: float = 5e-4
    q_target_update_rate: int = 1000
    batch_size: int = 64
    use_bias: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

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


def save_output(config: TrainConfig, metrics, q_network):
    model_filepath = os.path.join('outputs', config.run_name + '.pt')
    torch.save(q_network.state_dict(), model_filepath)

    output_data = {
        "args": config.to_dict(),
        "metrics": metrics,
    }
    results_filepath = os.path.join('outputs', config.run_name + '.json')
    with open(results_filepath, "w") as f:
        json.dump(output_data, f, indent=4)


def train(config: TrainConfig):
    # env
    reward_method = getattr(BFPRewardMethod, config.reward_method)
    env = BFP_ENV(config.N, reward_method, config.episode_length_factor)
    
    # model
    q_network = BFP_DQN(config.N, config.hidden_dim, use_bias=config.use_bias)

    # behavior collector
    epsilon_model = partial(decaying_epsilon, epsilon_high=config.epsilon_high, epsilon_low=config.epsilon_low, epsilon_decay=config.epsilon_decay)
    behavior_policy = partial(epsilon_greedy_policy, q_network=q_network, epsilon_model=epsilon_model)
    experience_replay = ExperienceReplay(env, behavior_policy, config.episodes_per_epoch)

    # optimizer
    optimizer = torch.optim.Adam(q_network.parameters(), lr=config.lr)

    # evaluator
    policy = partial(epsilon_greedy_policy, q_network=q_network, epsilon_model=lambda epoch: 0, epoch=0)
    evaluator = partial(collect_by_policy, env, policy, episodes_per_epoch=5)

    metrics = training_loop(
        experience_replay,
        optimizer,
        q_network,
        evaluator,
        config.epochs,
        config.batch_size,
        config.p_her,
        config.q_target_update_rate,
        config.gamma
    )

    save_output(config, metrics, q_network)


if __name__ == "__main__":
    config = simple_parsing.parse(TrainConfig)
    train(config)
