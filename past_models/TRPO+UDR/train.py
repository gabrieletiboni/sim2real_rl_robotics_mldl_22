from email import policy
from gc import callbacks
from math import gamma
from operator import contains
from unittest.mock import Base
import torch
import gym
import argparse
from sb3_contrib import TRPO

from env.custom_hopper import *


def main():

    # torch.cuda.set_per_process_memory_fraction(0.7, 0)

    env = gym.make("CustomHopper-udr-v0")
    # env = gym.make('CustomHopper-target-v0')

    print("Action space:", env.action_space)
    print("State space:", env.observation_space)
    print("Dynamics parameters:", env.get_parameters())

    """
		Training
	"""

    model = TRPO('MlpPolicy', env, verbose = 1)

    model.learn(total_timesteps = 500000)

    model.save("trpo_udr_model.mdl")


if __name__ == "__main__":
    main()
