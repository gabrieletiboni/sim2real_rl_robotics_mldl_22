"""Test an RL agent on the OpenAI Gym Hopper environment"""

import torch
import gym
import argparse
from sb3_contrib import TRPO
import numpy as np
from env.custom_hopper import *
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=False)
    parser.add_argument("--render", default=False, action="store_true", help="Render the simulator")
    return parser.parse_args()


args = parse_args()


def main():

    if args.env == False:
        env = gym.make("CustomHopper-source-v0")
    elif args.env == 'udr':
        env = gym.make('CustomHopper-udr-v0')
    elif args.env == 'target':
        env = gym.make('CustomHopper-target-v0')
        
    fnames = []
    for file in glob.glob("to_test/*"):
        fnames.append(file)
        
    for modelname in fnames:
        obs = env.reset()
        totalreward = 0
        allrewards = []
        model = TRPO.load(modelname, env)
        
        for i in range(100):
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
                 
        with open("results.txt", "a") as f:  
            f.write(f"{modelname.replace('-',',').replace('.mdl','')},{np.mean(allrewards)},{np.sqrt(np.var(allrewards))}\n") 
            print(f"Model: {modelname}\n")
            print(f"Average reward: {np.mean(allrewards)}\n")
            print(f"Average reward stdev: {np.sqrt(np.var(allrewards))}\n")

if __name__ == "__main__":
    main()
