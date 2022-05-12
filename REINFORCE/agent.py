import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        # Learned standard deviation for exploration at training time
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        """
            Critic network
        """
        # TODO 2.2.b: critic network for actor-critic algorithm

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        """
            Critic
        """
        # TODO 2.2.b: forward in the critic network

        return normal_dist


# The baselineNet used to compute a baseline for the REINFORCE algorithm
class Baseline(torch.nn.Module):
    def __init__(self, state_size, hidden_size=256):
        super().__init__()
        self.dense_layer_1 = torch.nn.Linear(state_size, hidden_size)
        self.dense_layer_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.clamp(x, -1.1, 1.1)
        x = F.relu(self.dense_layer_1(x))
        x = F.relu(self.dense_layer_2(x))
        return self.output(x).squeeze(1)


class Agent(object):
    def __init__(self, policy, baseline, device="cpu"):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.baseline = baseline.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.bsoptimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        action_log_probs = (
            torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        )
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = (
            torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        )
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        #
        # TODO 2.2.a:
        #             - compute discounted returns
        #             - compute policy gradient loss function given actions and returns
        #             - compute gradients and step the optimizer
        #

        # Discounted Returns
        discounted_rewards = []

        for t in range(len(rewards)):
            d_r = 0
            pw = 0
            for r_ in rewards[t:]:
                d_r = d_r + self.gamma**pw * r_
                pw = pw + 1
            discounted_rewards.append(d_r)

        discounted_rewards = torch.tensor(
            discounted_rewards, dtype=torch.float32, device=self.train_device
        )

        # REINFORCE LOSS
        # `discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) / (torch.std(discounted_rewards))
        # `policy_gradient = -action_log_probs * discounted_rewards
        # self.policy.zero_grad()
        # policy_gradient.sum().backward()
        # self.optimizer.step()
        
        #Report both on the final table

        # REINFORCE BASELINE LOSS
        dsr_estimates = self.baseline(states).to(self.train_device)

        with torch.no_grad():
            advantages = discounted_rewards - dsr_estimates

        actor_loss = torch.mean(-action_log_probs * advantages)
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        baseline_loss_fn = torch.nn.MSELoss()
        baseline_loss = baseline_loss_fn(dsr_estimates, discounted_rewards)
        self.bsoptimizer.zero_grad()
        baseline_loss.backward()
        self.bsoptimizer.step()

        #
        # TODO 2.2.b:
        #             - compute boostrapped discounted return estimates
        #             - compute advantage terms
        #             - compute actor loss and critic loss
        #             - compute gradients and step the optimizer
        #

        return

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:  # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

    def reset_outcomes(self):
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
