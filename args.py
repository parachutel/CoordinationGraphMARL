import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Meta
    parser.add_argument('--name',
                        '-n',
                        type=str,
                        help='Name to identify training or test run.')
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='GPU ID')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')
    parser.add_argument('--enforce_cpu',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether forcing CPU.')
    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')
    parser.add_argument('--debug',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether print debug info.')
    parser.add_argument('--record',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether record videos for evaluation.')
    parser.add_argument('--video_save_path',
                        type=str,
                        default='./save/video_recording/',
                        help='Path to load as a model checkpoint.')


    # Train
    parser.add_argument('--n_epochs',
                        type=int,
                        default=1000,
                        help='Number of epochs for which to train. Negative means forever.')
    parser.add_argument('--policy_lr',
                        type=float,
                        default=3e-4, # 1e-4
                        help='Learning rate for policy.')
    parser.add_argument('--baseline_lr',
                        type=float,
                        default=3e-4,
                        help='Learning rate for baseline/value function.')
    parser.add_argument('--n_episodes_per_epoch',
                        type=int,
                        default=50,
                        help='Number of episodes in one epoch.')
    parser.add_argument('--n_eval_episodes',
                        type=int,
                        default=100,
                        help='Number of episodes in one evaluation.')
    parser.add_argument('--eval_greedy',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether use greedy action in evaluation mode.')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.01,
                        help='L2 regularization coefficient.')
    # parser.add_argument('--metric_name',
    #                     type=str,
    #                     default='<Num Achieved Individual Instructions>',
    #                     help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='[Not used] Maximum gradient norm for gradient clipping.')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed for reproducibility.')
    parser.add_argument('--visualize',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether visualize the animation of a policy.')
    parser.add_argument('--eval',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether enter eval mode.')
    parser.add_argument('--eval_steps',
                        type=int,
                        default=20000,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--eval_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs between successive evaluations.')
    parser.add_argument('--continuing_checkpoint',
                        type=str,
                        default=None,
                        help='Whether continue from a checkpoint.')

    # Env
    parser.add_argument('--env',
                        type=str,
                        default='traffic',
                        help='Which env to use.')
    parser.add_argument('--n_agents',
                        type=int,
                        default=8,
                        help='Number of agents')
    parser.add_argument('--proximity_threshold',
                        type=float,
                        default=1.5)
    parser.add_argument('--self_connected_adj',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether adding self-connection to adjacency matrix.')
    parser.add_argument('--inverse_D',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether inversing the sqrt of degree matrix.')
    parser.add_argument('--max_episode_steps',
                        type=int,
                        default=100,
                        help='The max number of steps in an episode.')
    parser.add_argument('--difficulty',
                        type=str,
                        default='hard',
                        help='[Traffic Junction] Game difficulty.')
    parser.add_argument('--penalty',
                        type=float,
                        default=0,
                        help='[Predator-Prey] Single-agent capture attempt penalty.')
    parser.add_argument('--n_entities',
                        type=int,
                        default=8,
                        help='[Predator-Prey] Number of prey.')
    parser.add_argument('--n_grids',
                        type=int,
                        default=10,
                        help='Size of grid world.')
    parser.add_argument('--agent_visible',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether other agents are visible to each agent.')
    parser.add_argument('--render_adj',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether draw edges during video rendering.')
    
    # Sampler
    parser.add_argument('--batch_size',
                        type=int,
                        default=60000,
                        help='[Num env steps] Batch size per GPU. Scales \
                              automatically when multiple GPUs are available.')
    parser.add_argument('--n_trajs_limit',
                        type=int,
                        default=400,
                        help='Number of trajectories.')
    parser.add_argument('--limit_by_traj',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether limit sampler iters using the number of trajs.')

    # Algo
    parser.add_argument('--discount',
                        type=float,
                        default=0.99,
                        help='Discount factor.')
    parser.add_argument('--gae_lambda',
                        type=float,
                        default=0.97,
                        help='Discount factor for TD(lambda) used by GAE.')
    parser.add_argument('--ent',
                        type=float,
                        default=0.05,
                        help='Policy entropy coefficient for policy gradient algorithm')
    parser.add_argument('--lr_clip_range',
                        type=float,
                        default=2e-1,
                        help='Likelihood ratio clipping range of PPO.')
    parser.add_argument('--optimization_n_minibatches',
                        type=int,
                        default=4,
                        help='Splitting trajectory samples into minibatches.')
    parser.add_argument('--optimization_mini_epochs',
                        type=int,
                        default=10,
                        help='Optimization epochs for each minibatch.')

#     parser.add_argument('--epsilon',
#                         type=float,
#                         default=1.0,
#                         help='The initial value of the coefficient for epsilon-greedy policy.')
#     parser.add_argument('--min_epsilon',
#                         type=float,
#                         default=0.1,
#                         help='The final minimum value of epsilon.')
#     parser.add_argument('--epsilon_decay',
#                         type=float,
#                         default=0.992,
#                         help='The decay rate of epsilon.')
#     parser.add_argument('--epsilon_decay_ratio',
#                         type=float,
#                         default=0.6,
#                         help='The ratio of epochs before decreasing to min epsilon.')
#     parser.add_argument('--tau',
#                         type=float,
#                         default=0.95,
#                         help='Weight coefficient for moving average update for target network.')
#     parser.add_argument('--update_target_by_copy',
#                         type=lambda s: s.lower().startswith('t'),
#                         default=False,
#                         help='Whether upadte target network by copying the current Q network,\
#                               otherwise use moving average with the current Q network.')
#     parser.add_argument('--replay_buffer_size',
#                         type=int,
#                         default=2e6,
#                         help='Number of transition tuples in the experience\
#                               replay buffer.')
#     parser.add_argument('--qf_update_steps',
#                         type=int,
#                         default=100,
#                         help='Update Q network how many times for one episode.')

    # Policy/Model
    parser.add_argument('--policy',
                        type=str,
                        default='proximal_cg',
                        help='Which policy to use.')
    parser.add_argument('--n_gcn_layers',
                        type=int,
                        default=2,
                        help='Number of GCN layers.')

    # Edge Predictor
    parser.add_argument('--edge_predictor_pred_method',
                        type=str,
                        default='gumbel',
                        help='How to predict discrete edges.')
    parser.add_argument('--edge_predictor_input_method',
                        type=str,
                        default='diff',
                        help='How to build inputs for edge predictor.')
    parser.add_argument('--edge_predictor_checkpoint_path',
                        type=str,
                        default='./save/meet/edge_pred_data/meet_14_input=diff_'
                            + 'threshold=2.0_nagents=10_npos=3e+06_neg2pos=2.0_'
                            + 'sc=False_weighdacay=0.00_tau=0.25-01/best.pth.tar')
    parser.add_argument('--tau',
                        type=float,
                        default=0.25,
                        help='Temperature for Gumbel softmax.')
    parser.add_argument('--max_n_positive_samples',
                        type=int,
                        default=3e6,
                        help='The maximum number of positive samples in training dataset.')
    parser.add_argument('--neg2pos',
                        type=float,
                        default=2.0,
                        help='The ratio between the number of negative samples and positive samples.')

    args = parser.parse_args()
    args.maximize_metric = True
    return args

args = get_args()