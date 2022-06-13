import argparse
import os
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
import TD3
import utils
from agent import *

def product(iterable: Iterable[int]) -> int:
    """Return the product of every element in iterable."""
    return reduce(operator.mul, iterable, 1)

# def main(
#     domain: str = 'walker',
#     task: str = 'run',
#     seed: int = 0,
# ):
#     ## hyper parameter
#     gamma = 0.99
#     num_episodes = 100
#     learning_rate = 0.001

#     torch.manual_seed(seed)
#     writer = SummaryWriter()
#     env: Environment = suite.load(domain, task, task_kwargs={'random': seed})
#     observation_spec = env.observation_spec()
#     observation_shape = sum(product(value.shape)
#                             for value in observation_spec.values())
#     action_spec = env.action_spec()
#     action_shape = product(action_spec.shape)
#     max_action = action_spec.maximum
#     min_action = action_spec.minimum
#     loc_action = (max_action + min_action) / 2
#     scale_action = (max_action - min_action) / 2

#     def map_action(input_: np.ndarray) -> np.ndarray:
#         return np.tanh(input_) * scale_action + loc_action

#     policy = Policy()
#     policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
#     critic = Critic()
#     critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
#     policy.train()
#     critic.train()

#     # def test_function(episode: int):
#     #     test(domain=domain,
#     #          episode=episode,
#     #          map_action=map_action,
#     #          policy=policy,
#     #          repeat=test_num,
#     #          seed=seed,
#     #          task=task,
#     #          writer=writer)

#     train(
#         env=env,
#         policy=policy,
#         critic=critic,
#         policy_optimizer=policy_optimizer,
#         critic_optimizer=critic_optimizer,
#         gamma=gamma,
#         num_episodes=num_episodes,
#         map_action=map_action,
#         writer=writer
#     )

#     # if log:
#     #     torch.save(policy.state_dict(), save_path+'/model.pth')

# def train(
#     env: Environment,
#     policy: Policy,
#     critic: Critic,
#     policy_optimizer: optim.Adam,
#     critic_optimizer: optim.Adam,
#     gamma: float,
#     num_episodes: int, 
#     map_action: Callable[[np.ndarray], np.ndarray],
#     writer: Optional[SummaryWriter]
# ):
#     replay_buffer = Memory(
#         memory_size=1000000,
#         batch_size=10,
#         state_size=24,
#         action_size=6,
#     )
#     for episode in trange(num_episodes):
#         time_step = env.reset()
#         while not time_step.last():
#             observation = flatten_and_concat(time_step.observation)
#             action = policy.act(observation)
#             time_step = env.step(map_action(action))
#             replay_buffer.store_transition(
#                 state=observation,
#                 action=action,
#                 reward=time_step.reward,
#                 next_state=time_step.observation,
#                 done=time_step.last(),
#             )
#             #TODO:update policy
#             if replay_buffer.buffer_size > 1000:
#                 batch = replay_buffer.sample()
#                 # update
#                 curr_values = critic.eval(batch.states, batch.actions)
#                 target_values = batch.rewards + gamma * (1 - batch.dones) * critic.eval(batch.next_states, policy.act(batch.next_states))
#                 critic_loss = nn.MSELoss()(curr_values, target_values)
                
#                 critic_optimizer.zero_grad()
#                 critic_loss.backward()
#                 critic_optimizer.step()

#                 policy_loss = -critic.eval(batch.states, policy(batch.states))


# def update(minibatch: Dict, policy: Policy, critic: Critic):
#     rewards = minibatch.rewards
#     target_value = rewards +  

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def eval_policy(eval_env: Environment, policy: TD3, seed: int, map_action: Callable[[np.ndarray], np.ndarray], eval_episodes=10):
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        time_step = eval_env.reset()
        while not time_step.last():
            state = flatten_and_concat(time_step.observation)
            action = policy.select_action(state)
            time_step_new = eval_env.step(map_action(action))
            avg_reward += time_step_new.reward
            time_step = time_step_new

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def main(args):
    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env: Environment = suite.load(args.env, args.task, task_kwargs={'random': args.seed})
    env_eval: Environment = suite.load(args.env, args.task, task_kwargs={'random': args.seed})

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_spec = env.observation_spec()
    state_dim = sum(product(value.shape)
                            for value in state_spec.values())
    action_spec = env.action_spec()
    action_dim = product(action_spec.shape)
    max_action = action_spec.maximum
    min_action = action_spec.minimum
    loc_action = (max_action + min_action) / 2
    scale_action = (max_action - min_action) / 2
    
    def map_action(input_: np.ndarray) -> np.ndarray:
        return np.tanh(input_) * scale_action + loc_action
    def get_random_action() -> np.ndarray:
        return np.random.uniform(min_action, max_action)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed, map_action)]

    time_step = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in trange(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        state = flatten_and_concat(time_step.observation)
        if t < args.start_timesteps:
            action = get_random_action()
        else:
            action = (
                policy.select_action(state)
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        time_step_new = env.step(action) 
        done_bool = float(time_step_new.last()) if episode_timesteps < env._max_episode_steps else 0
        next_state = flatten_and_concat(time_step_new.observation)
        reward = time_step_new.reward

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if time_step_new.last(): 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            time_step = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
    
if __name__ == "__main__":
	
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="walker")  # dm_control environment name
    parser.add_argument("--task", default="run")    # task name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    main(args)
