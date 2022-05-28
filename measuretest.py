from env.custom_hopper import CustomHopper
from scipy.stats import truncnorm
import gym

env = gym.make("CustomHopper-bayrn-v0")
customhopper = env.

for i in range(100):
    print(customhopper.get_parameters())
    customhopper.reset_model()