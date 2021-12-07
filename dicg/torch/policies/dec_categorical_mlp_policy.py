"""CategoricalMLPPolicy."""
import akro
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from dicg.torch.modules import CategoricalMLPModule
from garage.torch.policies import Policy


class DecCategoricalMLPPolicy(CategoricalMLPModule):

    def __init__(self,
                 env_spec,
                 n_agents,
                 hidden_sizes=(128, 64, 32),
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='DecCategoricalMLPPolicy',
                 device='cpu'):

        self.centralized = True # centralized training

        self._n_agents = n_agents
        self._obs_dim = int(env_spec.observation_space.flat_dim / n_agents) # dec obs_dim
        self._action_dim = env_spec.action_space.n

        CategoricalMLPModule.__init__(self,
                                      input_dim=self._obs_dim,
                                      output_dim=self._action_dim,
                                      hidden_sizes=hidden_sizes,
                                      hidden_nonlinearity=hidden_nonlinearity,
                                      hidden_w_init=hidden_w_init,
                                      hidden_b_init=hidden_b_init,
                                      output_nonlinearity=output_nonlinearity,
                                      output_w_init=output_w_init,
                                      output_b_init=output_b_init,
                                      layer_normalization=layer_normalization)
        self.proximity_adj = False
        self.device = device

    def grad_norm(self):
        return np.sqrt(
            np.sum([p.grad.norm(2).item() ** 2 for p in self.parameters()]))

    def forward(self, obs, avail_actions, adjs=None):
        if not torch.is_tensor(obs):
            obs = torch.Tensor(obs).to(self.device)
        obs = obs.reshape(obs.shape[:-1] + (self._n_agents, -1))
        dist = super().forward(obs)
        # Apply available actions mask
        avail_actions = avail_actions.reshape(
            avail_actions.shape[:-1] + (self._n_agents, -1))
        masked_probs = dist.probs * torch.Tensor(avail_actions).to(self.device) # mask
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True) # renormalize
        masked_dist = Categorical(probs=masked_probs) # redefine distribution
        return masked_dist, adjs

    def get_actions(self, observations, avail_actions, adjs=None, alive_masks=None, greedy=False):
        with torch.no_grad():
            # obs.shape = (n_agents, n_envs, obs_dim)
            dist, _ = self.forward(observations, avail_actions)
            if not greedy:
                actions = dist.sample().cpu().numpy()
            else:
                actions = np.argmax(dist.probs.cpu().numpy(), axis=-1)
            agent_infos = {}
            agent_infos['action_probs'] = [dist.probs[i].cpu().numpy() 
                for i in range(len(actions))]
            return actions, agent_infos

    def log_likelihood(self, observations, avail_actions, actions, adjs=None, alive_masks=None):
        dist, _ = self.forward(observations, avail_actions)
        # For n agents, action probabilities are treated as independent
        # Pa = prob_i^n Pa_i
        # => log(Pa) = sum_i^n log(Pa_i)
        return dist.log_prob(actions).sum(axis=-1)

    def entropy(self, observations, avail_actions, adjs=None, alive_masks=None):
        dist, _ = self.forward(observations, avail_actions)
        return dist.entropy().mean(axis=-1)

    def reset(self, dones=None):
        pass

    @property
    def recurrent(self):
        return False