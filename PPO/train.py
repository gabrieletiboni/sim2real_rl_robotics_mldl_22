from email import policy
from operator import contains
from unittest.mock import Base
import torch
import gym
import argparse
from stable_baselines3 import PPO

from env.custom_hopper import *
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-episodes", default=100000, type=int, help="Number of training episodes"
    )
    parser.add_argument(
        "--print-every", default=1000, type=int, help="Print info every <> episodes"
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="network device [cpu, cuda]"
    )

    return parser.parse_args()


args = parse_args()


def main():

    # torch.cuda.set_per_process_memory_fraction(0.7, 0)

    env = gym.make("CustomHopper-source-v0")
    # env = gym.make('CustomHopper-target-v0')

    print("Action space:", env.action_space)
    print("State space:", env.observation_space)
    print("Dynamics parameters:", env.get_parameters())

    """
		Training
	"""

    model = PPO('MlpPolicy', env)

    model.learn(total_timesteps = 100000)

    model.save("model.mdl")


if __name__ == "__main__":
    main()
