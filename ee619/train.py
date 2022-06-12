import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from functools import reduce
import operator
from typing import Callable, Iterable, List, NamedTuple, Optional, Dict
from dm_control import suite
from dm_control.rl.control import Environment
from tqdm import trange
from agent import Policy, Critic, flatten_and_concat, Memory

def product(iterable: Iterable[int]) -> int:
    """Return the product of every element in iterable."""
    return reduce(operator.mul, iterable, 1)

def main(
    domain: str = 'walker',
    task: str = 'run',
    seed: int = 0,
):
    ## hyper parameter
    gamma = 0.99
    num_episodes = 100
    learning_rate = 0.001

    torch.manual_seed(seed)
    writer = SummaryWriter()
    env: Environment = suite.load(domain, task, task_kwargs={'random': seed})
    observation_spec = env.observation_spec()
    observation_shape = sum(product(value.shape)
                            for value in observation_spec.values())
    action_spec = env.action_spec()
    action_shape = product(action_spec.shape)
    max_action = action_spec.maximum
    min_action = action_spec.minimum
    loc_action = (max_action + min_action) / 2
    scale_action = (max_action - min_action) / 2

    def map_action(input_: np.ndarray) -> np.ndarray:
        return np.tanh(input_) * scale_action + loc_action

    policy = Policy()
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    critic = Critic()
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
    policy.train()
    critic.train()

    # def test_function(episode: int):
    #     test(domain=domain,
    #          episode=episode,
    #          map_action=map_action,
    #          policy=policy,
    #          repeat=test_num,
    #          seed=seed,
    #          task=task,
    #          writer=writer)

    train(
        env=env,
        policy=policy,
        critic=critic,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        gamma=gamma,
        num_episodes=num_episodes,
        map_action=map_action,
        writer=writer
    )

    # if log:
    #     torch.save(policy.state_dict(), save_path+'/model.pth')

def train(
    env: Environment,
    policy: Policy,
    critic: Critic,
    policy_optimizer: optim.Adam,
    critic_optimizer: optim.Adam,
    gamma: float,
    num_episodes: int, 
    map_action: Callable[[np.ndarray], np.ndarray],
    writer: Optional[SummaryWriter]
):
    replay_buffer = Memory(
        memory_size=1000000,
        batch_size=10,
        state_size=24,
        action_size=6,
    )
    for episode in trange(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            observation = flatten_and_concat(time_step.observation)
            action = policy.act(observation)
            time_step = env.step(map_action(action))
            replay_buffer.store_transition(
                state=observation,
                action=action,
                reward=time_step.reward,
                next_state=time_step.observation,
                done=time_step.last(),
            )
            #TODO:update policy
            if replay_buffer.buffer_size > 1000:
                batch = replay_buffer.sample()
                # update
                curr_values = critic.eval(batch.states, batch.actions)
                target_values = batch.rewards + gamma * (1 - batch.dones) * critic.eval(batch.next_states, policy.act(batch.next_states))
                critic_loss = nn.MSELoss()(curr_values, target_values)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                policy_loss = -critic.eval(batch.states, policy(batch.states))


def update(minibatch: Dict, policy: Policy, critic: Critic):
    rewards = minibatch.rewards
    target_value = rewards +  

if __name__ == "__main__":
    main()