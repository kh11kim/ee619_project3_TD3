"""Agent for DMControl Walker-Run task."""
from os.path import abspath, dirname, realpath
from typing import Dict, Tuple

from dm_env import TimeStep
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple

ROOT = dirname(abspath(realpath(__file__)))  # path to the ee619 directory
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def flatten_and_concat(dmc_observation: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert a DMControl observation (OrderedDict of NumPy arrays)
    into a single NumPy array.

    """
    return np.concatenate([[obs] if np.isscalar(obs) else obs.ravel()
                           for obs in dmc_observation.values()])
def to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert NumPy array to a PyTorch Tensor of data type torch.float32"""
    return torch.as_tensor(array, dtype=torch.float32).to(device)

class Agent:
    """Agent for a Walker2DBullet environment."""
    def __init__(self) -> None:
        # Create class variables here if you need to.
        # Example:
        #     self.policy = torch.nn.Sequential(...)
        pass

    def act(self, time_step: TimeStep) -> np.ndarray:
        """Returns the action to take for the current time-step.

        Args:
            time_step: a namedtuple with four fields step_type, reward,
                discount, and observation.
        """
        # You can access each member of time_step by time_step.[name], a
        # for example, time_step.reward or time_step.observation.
        # You may also check if the current time-step is the last one
        # of the episode, by calling the method time_step.last().
        # The return value will be True if it is the last time-step,
        # and False otherwise.
        # Note that the observation is given as an OrderedDict of
        # NumPy arrays, so you would need to convert it into a
        # single NumPy array before you feed it into your network.
        # It can be done by using the `flatten_and_concat` function, e.g.,
        #   observation = flatten_and_concat(time_step.observation)
        #   logits = self.policy(torch.as_tensor(observation))
        return np.ones(6)

    def load(self):
        """Loads network parameters if there are any."""
        # Example:
        #   path = join(ROOT, 'model.pth')
        #   self.policy.load_state_dict(torch.load(path))

class Net(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_layers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, out_features)
        )
    
    def forward(self, x: torch.tensor) -> np.ndarray:
        return self.net(x)

class Policy(Net):
    def __init__(self):
        super().__init__(
            in_features=24,
            out_features=6,
            hidden_layers=64
        )
        self.to(device)

    def act(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        action = self(to_tensor(observation).unsqueeze(0))
        return action.squeeze(0).cpu().numpy()

class Critic:
    def __init__(self):
        self.net = Net(30, 1, 64).to(device)
    
    def eval(
        self, 
        state: np.ndarray, 
        action: np.ndarray,
    ):
        state = to_tensor(state)
        action = to_tensor(action)
        x = torch.concat(state, action)
        return self.net(x)

class Memory:
    def __init__(
        self, 
        memory_size: int,
        batch_size: int,
        state_size: int,
        action_size: int,
    ):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.states = np.zeros(memory_size, state_size)
        self.actions = np.zeros(memory_size, action_size)
        self.rewards = np.zeros(memory_size, 1)
        self.next_states = np.zeros(memory_size, state_size)
        self.dones = np.zeros(memory_size, 1, dtype=bool)
        
        self.index = 0
        self.buffer_size = 0
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.states[self.index, :] = state
        self.actions[self.index, :] = action
        self.rewards[self.index, :] = reward
        self.next_states[self.index, :] = next_state
        self.dones[self.index, :] = done
        self.index = (self.index + 1) % self.memory_size
        self.buffer_size = self.buffer_size + 1 if self.buffer_size + 1 <= self.memory_size else self.memory_size

    def sample(self):
        indices = np.random.choice(range(self.buffer_size), self.batch_size)
        minibatch = dict(
            states=self.states[indices, :],
            actions=self.actions[indices, :],
            rewards=self.rewards[indices, :],
            next_states=self.next_states[indices, :],
            dones=self.dones[indices, :]
        )
        return minibatch