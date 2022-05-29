from env.custom_hopper import CustomHopper
from scipy.stats import truncnorm
import gym
import torch
import numpy as np

env = gym.make("CustomHopper-source-v0")

tensor0 = torch.empty((1,6), dtype=torch.double)
tensor = torch.tensor([[1,1,1,1,1,1]], dtype=torch.double)
tensor2 = torch.tensor([1,1,2,1,1,1], dtype=torch.double)

tensor = torch.cat((tensor0, tensor), dim=0)
tensor = torch.cat((tensor, tensor2), dim=0)

print(tensor)

a = list(tensor.numpy())

thigha, thighb, lega, legb, foota, footb = a[0]