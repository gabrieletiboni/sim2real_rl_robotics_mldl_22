import gym
from sb3_contrib import TRPO
import botorch
import torch
from env.custom_hopper import *
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import gpytorch

def target_domain_return(n, target_env, policy):
    env = target_env
    obs = env.reset()
    allrewards = []
    totalreward = 0

    for _ in range(n):
        done = False

        while not done:
            action, _states = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            totalreward = totalreward+reward

            if done:
                obs = env.reset()
                allrewards.append(totalreward)
                totalreward = 0
    
    return np.mean(allrewards)


class DomainParametersDistributions():
    def __init__(self, env: gym.Env) -> None:
        self.torso_mass = env.env.get_parameters()[0]
        self.thigha, self.thighb = env.env.get_parameters()[1]-1, env.env.get_parameters()[1]+1
        self.lega, self.legb = env.env.get_parameters()[2]-1, env.env.get_parameters()[2]-1+1
        self.foota, self.footb = env.env.get_parameters()[3]-1, env.env.get_parameters()[3]+1
        self.jreals = torch.empty((1,1), dtype=torch.double)

    def initialization_phase(self, n_params):
        self.domain_distr_param = torch.empty((1,6), dtype=torch.double)
        env_params = []
        ab_norm = truncnorm(-1,1)

        for _ in range(n_params):

            thigha, thighb = self.thigha+ab_norm.rvs(), self.thighb+ab_norm.rvs()
            lega, legb = self.lega+ab_norm.rvs(), self.legb+ab_norm.rvs()
            foota, footb = self.foota+ab_norm.rvs(), self.footb+ab_norm.rvs()

            thighval = uniform(thigha, thighb).rvs()
            legval = uniform(lega, legb).rvs()
            footval = uniform(foota, footb).rvs()

            torch.cat((self.domain_distr_param, torch.tensor([[thigha, thighb, lega, legb, foota, footb]], dtype=torch.double)), dim=0)
            env_params.append(np.array([self.torso_mass, thighval, legval, footval], dtype=np.float32))

        return env_params

    def fit_gaussian(self):
        self.gp = botorch.models.SingleTaskGP(self.domain_distr_param, self.jreals) #Uses a matern kernel by default
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        botorch.fit.fit_gpytorch_model(self.mll)
        #Acquisition function

        self.UCB = botorch.acquisition.UpperConfidenceBound(self.gp, beta = 0.1)

    def optimize_acq(self):
        # Source domain base params [2.53429174 3.92699082 2.71433605 5.0893801 ]
        
        bounds = torch.tensor([[2.,4.,1.,3.,2.,5.],[4.,8.,2.5,5.,4.,7.]], dtype=torch.double)
        
        #bounds = torch.tensor([[0.25,0.25,0.25,0.25,0.25,0.25],[10,10,10,10,10,10]], dtype=torch.double)

        candidate, _ = botorch.optim.optimize_acqf(self.UCB, bounds=bounds, q=1, num_restarts=200, raw_samples=512)
        torch.cat((self.domain_distr_param,candidate), dim=0)

        return list(candidate.numpy()[0])

        # The output is tensor([[2.6634, 8.2721, 1.9408, 5.6605, 1.5373, 6.8326]]) so a list of ranges in the format a,b,a,b,a,b

    def update_jreal(self, jreal):
        torch.cat((self.jreals,torch.tensor([[jreal]], dtype=torch.double)),dim=0)

    def get_env_params(self):

        thigha, thighb, lega, legb, foota, footb = self.optimize_acq()
        thighval = uniform(thigha, thighb).rvs()
        legval = uniform(lega, legb).rvs()
        footval = uniform(foota, footb).rvs()

        return [self.torso_mass, thighval, legval, footval]



# BayRN implementation
# Basically we want to make three steps, but I need a way to update the environment variables using the sb3_contrib method, i could implement my own learn thing i guess, i have to check
# I want to use truncated normal distributions, i will have 3 distributions with 2 parameters each
# to access my env variables i just need to call the env.env method
# remember to do the initialization phase first, in this one i might use udr
# thigh, leg and foot

# TODO
# Test different hyperparameters and bounds
# Implement other things if we want
# test 
# Go on stable-baseline webside and implement a custom callback function to do some logging
# Use callbacks to save the best performiong model



def main():

    env = gym.make("CustomHopper-source-v0")
    target_env = gym.make("CustomHopper-target-v0")

    domain_dist_params = DomainParametersDistributions(env)

    print("Action space:", env.action_space)
    print("State space:", env.observation_space)
    print("Dynamics parameters:", env.get_parameters())

    torch.device("cpu")

    """
		Training
	"""

    policy = TRPO('MlpPolicy', env, verbose = 0)

    #Initialization
    init_steps = 10
    init_env_params = domain_dist_params.initialization_phase(init_steps)

    for i in range(init_steps):
        env.env.set_parameters(*init_env_params[i])
        policy.learn(total_timesteps = 10000)
        Jreal = target_domain_return(n=5, target_env=target_env, policy=policy)
        domain_dist_params.update_jreal(Jreal)
        print(f"Initial sampling {i+1}")

    domain_dist_params.fit_gaussian()

    print("Begin Training")
    
    for _ in range(90):

        env.env.set_parameters(*domain_dist_params.get_env_params())

        policy.learn(total_timesteps = 10000)

        Jreal = target_domain_return(n=5, target_env=target_env, policy=policy)
        print(f"Return on real world for iteration {_+1}: {Jreal}")

        domain_dist_params.update_jreal(Jreal)
        domain_dist_params.fit_gaussian()


    policy.save("trpo_bayrn_model_v3.mdl")


if __name__ == "__main__":
    main()