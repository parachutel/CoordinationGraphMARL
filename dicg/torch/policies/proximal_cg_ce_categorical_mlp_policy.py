import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from dicg.torch.modules import CategoricalMLPModule, DICGBase

class ProximalCGCECategoricalMLPPolicy(DICGBase):
    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 n_gcn_layers=2,
                 gcn_bias=True,
                 categorical_mlp_hidden_sizes=(128, 64, 32),
                 name='proximal_cg_ce_categorical_mlp_policy',
                 device='cpu'):

        super().__init__(
            env_spec=env_spec,
            n_agents=n_agents,
            encoder_hidden_sizes=encoder_hidden_sizes,
            embedding_dim=embedding_dim,
            n_gcn_layers=n_gcn_layers,
            gcn_bias=gcn_bias,
            name=name
        )
        
        # Policy layer
        self.categorical_output_layer = \
            CategoricalMLPModule(input_dim=self._embedding_dim * 2,
                                 output_dim=self._action_dim,
                                 hidden_sizes=categorical_mlp_hidden_sizes,
                                 hidden_nonlinearity=F.relu)

        self.proximity_adj = True
        self.device = device

    def forward(self, obs_n, avail_actions_n, adjs, alive_masks):
        '''
            Adjacency matrices are always obtained from env/outside algo
        '''

        if not torch.is_tensor(obs_n):
            obs_n = torch.Tensor(obs_n).to(self.device)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        
        if self.proximity_adj:
            assert adjs is not None
            assert alive_masks is None
        if self.proximity_adj and not torch.is_tensor(adjs):
            adjs = torch.Tensor(adjs).to(self.device)
        if not self.proximity_adj and not torch.is_tensor(alive_masks):
            alive_masks = torch.Tensor(alive_masks).float().to(self.device)
        
        if self.proximity_adj and len(adjs.shape) > 2:
            # flattened by sampler
            # restore shape during training
            adjs = adjs.reshape(adjs.shape[:-1] + (self._n_agents, -1))

        avail_actions_n = avail_actions_n.reshape(
            avail_actions_n.shape[:-1] + (self._n_agents, -1))
        
        embeddings_collection, _ = super().forward(obs_n, adjs=adjs, 
            alive_masks=alive_masks)

        # (n_paths, max_path_length, n_agents, action_space_dim)
        # or (n_agents, action_space_dim)
        embeddings_concat = torch.cat(
            [embeddings_collection[0], embeddings_collection[-1]], dim=-1)
        dists_n = self.categorical_output_layer.forward(embeddings_concat)

        # Apply available actions mask
        masked_probs = dists_n.probs * torch.Tensor(avail_actions_n).to(self.device) # mask
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True) # renormalize
        masked_dists_n = Categorical(probs=masked_probs) # redefine distribution

        return masked_dists_n, adjs

    def get_actions(self, obs_n, avail_actions_n, adjs=None, alive_masks=None, greedy=False):
        if self.proximity_adj:
            assert adjs is not None
        else:
            assert alive_masks is not None
        with torch.no_grad():
            dists_n, adjs = self.forward(obs_n, avail_actions_n, adjs, alive_masks)
            if not greedy:
                actions_n = dists_n.sample().cpu().numpy()
            else:
                actions_n = np.argmax(dists_n.probs.cpu().numpy(), axis=-1)
            agent_infos_n = {}
            agent_infos_n['action_probs'] = [dists_n.probs[i].cpu().numpy() 
                for i in range(len(actions_n))]
            if adjs is not None:
                agent_infos_n['adjs'] = adjs.cpu().numpy()

            return actions_n, agent_infos_n

    def entropy(self, observations, avail_actions, adjs=None, alive_masks=None):
        if self.proximity_adj:
            assert adjs is not None
        else:
            assert alive_masks is not None
        # print('obs.shape =', observations.shape)
        dists_n, _ = self.forward(observations, avail_actions, adjs, alive_masks)
        # print('dist =', dists_n)
        # print('dist.probs =', dists_n.probs)
        entropy = dists_n.entropy()
        # print('entropy.shapeBefore =', entropy.shape)
        entropy = entropy.mean(axis=-1) # Asuming independent actions
        # print('entropy.shapeAfter =', entropy.shape)
        return entropy

    def log_likelihood(self, observations, avail_actions, actions, adjs=None, alive_masks=None):
        if self.proximity_adj:
            assert adjs is not None
        else:
            assert alive_masks is not None
        dists_n, _ = self.forward(observations, avail_actions, adjs, alive_masks)
        llhs = dists_n.log_prob(actions)
        # llhs.shape = (n_paths, max_path_length, n_agents)
        # For n agents action probability can be treated as independent
        # Pa = prob_i^n Pa_i
        # log(Pa) = sum_i^n log(Pa_i)
        llhs = llhs.sum(axis=-1) # Asuming independent actions
        # llhs.shape = (n_paths, max_path_length)
        return llhs

    @property
    def recurrent(self):
        return False
