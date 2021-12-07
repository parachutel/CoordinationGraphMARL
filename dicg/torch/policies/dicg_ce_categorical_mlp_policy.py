from dicg.torch.policies.proximal_cg_ce_categorical_mlp_policy \
    import ProximalCGCECategoricalMLPPolicy

class DICGCECategoricalMLPPolicy(ProximalCGCECategoricalMLPPolicy):
    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 n_gcn_layers=2,
                 gcn_bias=True,
                 categorical_mlp_hidden_sizes=(128, 64, 32),
                 name='dicg_ce_categorical_mlp_policy',
                 device='cpu'):

        super().__init__(
            env_spec=env_spec,
            n_agents=n_agents,
            encoder_hidden_sizes=encoder_hidden_sizes,
            embedding_dim=embedding_dim,
            n_gcn_layers=n_gcn_layers,
            gcn_bias=gcn_bias,
            categorical_mlp_hidden_sizes=categorical_mlp_hidden_sizes,
            name=name,
            device=device
        )
        
        self.proximity_adj = False