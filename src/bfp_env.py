from dataclasses import dataclass
from copy import deepcopy
from enum import Enum

import torch

BFPState = torch.Tensor
BFPAction = int
BFPReward = float


@dataclass
class BFPAugState:
    state: BFPState
    goal: BFPState

    def augstate(self):
        return torch.cat([self.state, self.goal])

    def set_goal(self, goal):
        return BFPAugState(self.state, goal)

@dataclass
class BFPStateInfo:
    done: bool
    dist: float

@dataclass
class BFPEnvConfig:
    N: int
    reward_method: str = "BINARY_REWARD"
    episode_length_factor: int = 1

class BFPEnv:
    def __init__(self, N, reward_method, episode_length_factor):
        self._N = N
        self.reward_method = reward_method
        self.max_steps = episode_length_factor * N
        
        self._aug_state = None
        self.current_step = 0
        
    def reset(self, *, seed=None, options=None) -> tuple[BFPAugState, BFPStateInfo]:
        self.current_step = 0
        done = True
        while done:
            self._aug_state = self._generate_initial_aug_state(self._N)
            state_info = self._state_info(self._aug_state)
            done = state_info.done
        return self._aug_state, state_info

    def step(self, action: BFPAction) -> tuple[BFPAugState, BFPReward, BFPStateInfo]:
        self.current_step += 1
        next_aug_state = self._next_state(self._aug_state, action)
        reward = self._reward(self._aug_state, action, next_aug_state)
        state_info = self._state_info(next_aug_state)
        
        if self.current_step >= self.max_steps:
             state_info.done = True

        self._aug_state = next_aug_state

        return next_aug_state, reward, state_info

    @classmethod
    def _generate_initial_aug_state(cls, N) -> BFPAugState:
        goal = cls._generate_random_state(N)
        state = cls._generate_random_state(N)
        return BFPAugState(state, goal)

    @staticmethod
    def _generate_random_state(N) -> BFPState:
        return torch.randint(0, 2, (N,))

    @staticmethod
    def _next_state(aug_state: BFPAugState, action: BFPAction) -> BFPAugState:
        next_state = deepcopy(aug_state)
        next_state.state[action] = 1 - next_state.state[action]
        return next_state

    def _reward(self, aug_state: BFPAugState, action: BFPAction, next_aug_state: BFPAugState) -> BFPReward:
        return self.reward_method(next_aug_state)

    @classmethod
    def binary_reward(cls, next_aug_state):
        if cls.done(next_aug_state):
            return 0
        else:
            return -1

    @staticmethod
    def shaped_reward(next_aug_state):
        return -((next_aug_state.state - next_aug_state.goal) ** 2).sum()

    @classmethod
    def _state_info(cls, aug_state: BFPAugState) -> BFPStateInfo:
        return BFPStateInfo(
            done=cls.done(aug_state),
            dist=cls.dist(aug_state)
        )

    @staticmethod
    def dist(aug_state: BFPAugState) -> float:
        return len(aug_state.state)-sum(aug_state.state == aug_state.goal)

    @staticmethod
    def done(aug_state) -> bool:
        return all(aug_state.state == aug_state.goal)

    @staticmethod
    def episode_metrics(transitions):
        return {
            "total_return": sum([t.reward for t in transitions]),
            "min_dist": min([BFPEnv.dist(t.next_state) for t in transitions]),
            "final_dist": BFPEnv.dist(transitions[-1].next_state)
        }

class BFPRewardMethod(Enum):
    BINARY_REWARD = BFPEnv.binary_reward
    SHAPED_REWARD = BFPEnv.shaped_reward
