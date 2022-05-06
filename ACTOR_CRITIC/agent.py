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
        self.fc1_critic = torch.nn.Linear(state_space, 256)
        self.fc2_critic = torch.nn.Linear(256, 256)
        self.output_critic = torch.nn.Linear(256, 1)

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
        x_critic = torch.clamp(x, -1.1, 1.1)
        x_critic = F.relu(self.fc1_critic(x_critic))
        x_critic = F.relu(self.fc2_critic(x_critic))
        value = self.output_critic(x_critic)

        return normal_dist, value


class Agent(object):
    def __init__(self, policy, device="cpu"):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.entropy_term = 0
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.values = []

    def update_policy(self):
        action_log_probs = (torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1))
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = (torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1))
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        values = (torch.stack(self.values, dim=0).to(self.train_device).squeeze(-1))

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

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=self.train_device)

        # REINFORCE LOSS
        # `discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) / (torch.std(discounted_rewards))
        # `policy_gradient = -action_log_probs * discounted_rewards
        # self.policy.zero_grad()
        # policy_gradient.sum().backward()
        # self.optimizer.step()
        
        # TODO 2.2.b:
        #             - compute boostrapped discounted return estimates
        #             - compute advantage terms
        #             - compute actor loss and critic loss
        #             - compute gradients and step the optimizer

        with torch.no_grad():
            advantages = discounted_rewards - values
            
        advantages = (advantages-torch.mean(advantages)+1e-12)/torch.std(advantages)

        actor_loss = torch.mean(-action_log_probs*advantages)
        critic_loss_fn = torch.nn.MSELoss()
        critic_loss = critic_loss_fn(values, discounted_rewards)
        ac_loss = actor_loss + critic_loss + 0.001 * self.entropy_term

        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()

        return

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, value = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:  # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()
            entropy = -torch.sum(normal_dist.mean*action_log_prob).detach().numpy()
            self.entropy_term += entropy

            return action, action_log_prob, value

    def store_outcome(self, state, next_state, action_log_prob, reward, done, value):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
        self.values.append(value)

    def reset_outcomes(self):
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.values = []



#TODO Domande da fare:
# - Cambia qualcosa imparare da uno state alla volta piuttosto che da tutti gli states assieme?
