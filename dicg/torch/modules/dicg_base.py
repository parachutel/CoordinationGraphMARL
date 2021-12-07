import akro
import torch
from torch import nn
import numpy as np

from torch.distributions import Categorical

from .mlp_encoder_module import MLPEncoderModule
from .attention_module import AttentionModule
from .graph_conv_module import GraphConvolutionModule

class DICGBase(nn.Module):
    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 attention_type='general',
                 n_gcn_layers=2,
                 gcn_bias=True,
                 state_include_actions=False,
                 proximity_adj=False,
                 name='dicg_base'):

        super().__init__()

        self.name = name

        self.dicg = True
        self.centralized = True
        self.vectorized = True
        self.proximity_adj = proximity_adj
        
        self._n_agents = n_agents
        self._cent_obs_dim = env_spec.observation_space.flat_dim
        self._dec_obs_dim = int(self._cent_obs_dim / n_agents)
        if isinstance(env_spec.action_space, akro.Discrete): 
            self._action_dim = env_spec.action_space.n
        else:
            self._action_dim = env_spec.action_space.shape[0]
        self._embedding_dim = embedding_dim # dev

        self.n_gcn_layers = n_gcn_layers # dev


        if state_include_actions:
            self._dec_obs_dim += self._action_dim
        
        self.encoder = MLPEncoderModule(input_dim=self._dec_obs_dim,
                                        output_dim=self._embedding_dim,
                                        hidden_sizes=encoder_hidden_sizes,
                                        output_nonlinearity=torch.nn.functional.relu)

        self.attention_layer = AttentionModule(dimensions=self._embedding_dim, 
                                               attention_type=attention_type)

        self.gcn_layers = nn.ModuleList([
            GraphConvolutionModule(in_features=self._embedding_dim, 
                                   out_features=self._embedding_dim, 
                                   bias=gcn_bias, 
                                   id=i) for i in range(self.n_gcn_layers)
        ])

    def grad_norm(self):
        return np.sqrt(
            np.sum([p.grad.norm(2).item() ** 2 for p in self.parameters() if p.grad is not None]))

    def forward(self, obs_n, adjs=None, alive_masks=None):
        # Partially decentralize, treating agents as being independent

        # (n_paths, max_path_length, n_agents, emb_feat_dim)
        # or (n_agents, emb_feat_dim)
        embeddings_collection = []
        gcn_in = self.encoder.forward(obs_n)
        embeddings_collection.append(gcn_in)

        # (n_paths, max_path_length, n_agents, n_agents)
        # or (n_agents, n_agents)
        if not self.proximity_adj:
            assert alive_masks is not None
            # strictly proximity based policy does not need alive_masks
            # DICG and other methods that learn graph need alive_masks
            adjs = self.attention_layer.forward(gcn_in, alive_masks=alive_masks)

        for i_layer, gcn_layer in enumerate(self.gcn_layers):
            # (n_paths, max_path_length, n_agents, emb_feat_dim)
            # or (n_agents, emb_feat_dim)
            gcn_out = gcn_layer.forward(gcn_in, adjs)
            gcn_in = gcn_in + gcn_out

        embeddings_collection.append(gcn_in)

        return embeddings_collection, adjs


    def reset(self, dones):
        pass

    

