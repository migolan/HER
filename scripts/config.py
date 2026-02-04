from dataclasses import dataclass, asdict
from typing import Optional

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
