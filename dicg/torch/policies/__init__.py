from dicg.torch.policies.dec_categorical_mlp_policy \
    import DecCategoricalMLPPolicy
from dicg.torch.policies.dicg_ce_categorical_mlp_policy \
    import DICGCECategoricalMLPPolicy
from dicg.torch.policies.proximal_cg_ce_categorical_mlp_policy \
    import ProximalCGCECategoricalMLPPolicy

__all__ = [
    'DecCategoricalMLPPolicy', 
    'DICGCECategoricalMLPPolicy',
    'ProximalCGCECategoricalMLPPolicy',
]
