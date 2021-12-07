from args import args
from garage.envs import GarageEnv
from dicg.torch.policies import *
from args import args

def make_mlp_policy(env, device):
    g_env = GarageEnv(env)
    if args.policy == 'proximal_cg':
        policy = ProximalCGCECategoricalMLPPolicy(
            env_spec=g_env.spec,
            n_agents=env.n_agents,
            n_gcn_layers=args.n_gcn_layers,
            device=device)
    
    elif args.policy == 'dicg_ce':
        policy = policy = DICGCECategoricalMLPPolicy(
            env_spec=g_env.spec,
            n_agents=env.n_agents,
            n_gcn_layers=args.n_gcn_layers,
            device=device)
    elif args.policy == 'de':
        policy = DecCategoricalMLPPolicy(
            env_spec=g_env, 
            n_agents=env.n_agents,
            device=device)
    else:
        NotImplementedError
    
    return policy
