import json
import os

import simple_parsing
import torch
import torch.nn.functional as F
from dataclasses import asdict, dataclass
from src.bfp_env import BFPEnvConfig, BFPEnv, BFPRewardMethod
from src.train_dqn import DQNTrainConfig, train_dqn


class SimpleDQN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_bias=True):
        super(SimpleDQN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim, bias=use_bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@dataclass
class BFP_DQN_ModelConfig:
    hidden_dim: int = 128
    use_bias: bool = True


class BFP_DQN(SimpleDQN):
    def __init__(self, state_dim, hidden_dim, use_bias=True):
        super(BFP_DQN, self).__init__(state_dim*2, hidden_dim, state_dim, use_bias=use_bias)

    def forward(self, x):
        x = [s.augstate().float() for s in x]
        x = torch.stack(x)
        return super(BFP_DQN, self).forward(x)


def get_env_and_model(
        env_config: BFPEnvConfig,
        model_config: BFP_DQN_ModelConfig
    ):
    # env
    reward_method = getattr(BFPRewardMethod, env_config.reward_method)
    env = BFPEnv(env_config.N, reward_method, env_config.episode_length_factor)
    
    # model
    q_network = BFP_DQN(env_config.N, model_config.hidden_dim, use_bias=model_config.use_bias)
    return env, q_network


def save_output(
        env_config: BFPEnvConfig,
        model_config: BFP_DQN_ModelConfig,
        train_config: DQNTrainConfig,
        metrics,
        q_network
    ):
    model_filepath = os.path.join('outputs', train_config.run_name + '.pt')
    torch.save(q_network.state_dict(), model_filepath)

    output_data = {
        "args": {
            "env": asdict(env_config),
            "model": asdict(model_config),
            "train": asdict(train_config)
        },
        "metrics": metrics,
    }
    results_filepath = os.path.join('outputs', train_config.run_name + '.json')
    with open(results_filepath, "w") as f:
        json.dump(output_data, f, indent=4)


def _parse_args():
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(BFPEnvConfig, dest="env")
    parser.add_arguments(BFP_DQN_ModelConfig, dest="model")
    parser.add_arguments(DQNTrainConfig, dest="train")
    args = parser.parse_args()
    return args.env, args.model, args.train


if __name__ == "__main__":
    env_config, model_config, train_config = _parse_args()
    env, q_network = get_env_and_model(env_config, model_config)
    metrics = train_dqn(train_config, env, q_network)
    save_output(env_config, model_config, train_config, metrics, q_network)