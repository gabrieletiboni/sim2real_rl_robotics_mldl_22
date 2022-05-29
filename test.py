"""Test an RL agent on the OpenAI Gym Hopper environment"""

import torch
import gym
import argparse
from sb3_contrib import TRPO
import numpy as np
from env.custom_hopper import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="Model path")
    parser.add_argument("--render", default=False, action="store_true", help="Render the simulator")
    parser.add_argument("--episodes", default=50, type=int, help="Number of test episodes")
    parser.add_argument("--targetenv", default=False, action="store_true")
    return parser.parse_args()


args = parse_args()


def main():

    if args.targetenv == False:
        env = gym.make("CustomHopper-source-v0")
    else:
        env = gym.make('CustomHopper-target-v0')

    obs = env.reset()
    totalreward = 0
    allrewards = []
    model = TRPO.load(args.model, env)
    
    for i in range(args.episodes):
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            totalreward = totalreward+reward

            if args.render == True:
                env.render()

            if done:
                obs = env.reset()
                allrewards.append(totalreward)
                totalreward = 0
                
    print(f"Average reward: {np.mean(allrewards)}")
    print(f"Average reward stdev: {np.sqrt(np.var(allrewards))}")

if __name__ == "__main__":
    main()
