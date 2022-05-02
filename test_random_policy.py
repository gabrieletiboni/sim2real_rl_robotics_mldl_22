"""Test a random policy on the OpenAI Gym Hopper environment

Play around with this code to get familiar with the
Hopper environment.

For example, what happens if you don't reset the environment
even after the episode is over?
When exactly is the episode over?
What is an action here?
"""

import gym
from env.custom_hopper import *


def main():

    env = gym.make('CustomHopper-source-v0')

    print('State space:', env.observation_space)
    print('Action space:', env.action_space)
    print('Dynamics parameters:', env.get_parameters())

    n_episodes = 500
    render = True

    for episode in range(n_episodes):
        done = False
        observation = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over

            action = env.action_space.sample()  # Sample random action

            # Step the simulator to the next timestep
            observation, reward, done, info = env.step(action)

            if render:
                env.render()


if __name__ == '__main__':
    main()
