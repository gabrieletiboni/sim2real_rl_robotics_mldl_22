"""Test an RL agent on the OpenAI Gym Hopper environment"""

import torch
import gym
import argparse
from stable_baselines3 import PPO

from env.custom_hopper import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="Model path")
    parser.add_argument(
        "--device", default="cpu", type=str, help="network device [cpu, cuda]"
    )
    parser.add_argument(
        "--render", default=False, action="store_true", help="Render the simulator"
    )
    parser.add_argument(
        "--episodes", default=10, type=int, help="Number of test episodes"
    )

    return parser.parse_args()


args = parse_args()


def main():

    #env = gym.make("CustomHopper-source-v0")
    env = gym.make('CustomHopper-target-v0')

    print("Action space:", env.action_space)
    print("State space:", env.observation_space)
    print("Dynamics parameters:", env.get_parameters())

    obs = env.reset()
    
    model = PPO.load(args.model, env)
    
    for i in range(args.episodes):
        
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if args.render == True:
            env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    main()
