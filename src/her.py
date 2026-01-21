from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import TypeVar, Generic
import random
import numpy as np

State = TypeVar("State")
Action = TypeVar("Action")
Reward = TypeVar("Reward")

@dataclass
class Transition(Generic[State, Action, Reward]):
    state: State
    action: Action
    reward: Reward
    next_state: State

def collect_by_policy(env, policy, episodes_per_epoch):
    epoch_episode_metrics = []
    epoch_transitions = []
    for _ in range(episodes_per_epoch):
        transitions, episode_metrics = generate_episode_transitions(env, policy)
        epoch_transitions.extend(transitions)
        epoch_episode_metrics.append(episode_metrics)
    epoch_episode_metrics = aggregate_epoch_episode_metrics(epoch_episode_metrics)
    return epoch_transitions, epoch_episode_metrics

def generate_episode_transitions(env, policy):
    transitions = []
    state, state_info = env.reset()
    while not state_info.done:
        action = policy(state=state)
        next_state, reward, state_info = env.step(action)
        transitions.append(Transition(state, action, reward, next_state))
        state = next_state

    episode_metrics = env.episode_metrics(transitions)
    return transitions, episode_metrics

def aggregate_epoch_episode_metrics(epoch_episode_metrics):
    agg = defaultdict(list)
    for ep in epoch_episode_metrics:
        for k, v in ep.items():
            agg[k].append(v)

    return {k: float(np.mean(v)) for k, v in agg.items()}

class ExperienceReplay:
    def __init__(self, env, behavior_policy, episodes_per_epoch):
        self.transitions = []
        self.env = env
        self.behavior_policy = behavior_policy
        self.episodes_per_epoch = episodes_per_epoch

    def collect_by_behavior_policy(self, epoch):
        behavior_policy_in_epoch = partial(self.behavior_policy, epoch=epoch)
        transitions, episode_metrics = collect_by_policy(self.env, behavior_policy_in_epoch, self.episodes_per_epoch)
        self.transitions.extend(transitions)
        return episode_metrics

    def sample(self, batch_size):
        return random.sample(self.transitions, min(len(self.transitions), batch_size))

    def her_sample(self, batch_size, p_her):
        n_orig = int(batch_size * (1 - p_her))
        n_her = batch_size - n_orig
        sampled_transitions = self.sample(batch_size=n_orig)
        flipped_transitions = self.sample(batch_size=n_her)
        new_transitions = []
        for t in flipped_transitions:
            new_state = t.state.set_goal(t.next_state.state)
            new_next_state = t.next_state.set_goal(t.next_state.state)
            new_reward = self.env._reward(new_state, t.action, new_next_state)
            new_transitions.append(Transition(state=new_state, action=t.action, reward=new_reward, next_state=new_next_state))
        return sampled_transitions + new_transitions
