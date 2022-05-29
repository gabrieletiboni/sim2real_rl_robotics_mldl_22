from env.custom_hopper import CustomHopper
from scipy.stats import truncnorm
import gym
import torch
import numpy as np

env = gym.make("CustomHopper-source-v0")

tensor = torch.from_numpy(np.array([[1,1,1,1,1,1]]))

a = list(tensor.numpy())

thigha, thighb, lega, legb, foota, footb = a[0]