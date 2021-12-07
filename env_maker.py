from envs.traffic_junction_wrapper import TrafficJunctionWrapper
from envs.predatorprey_wrapper import PredatorPreyWrapper
from envs.meet_wrapper import MeetWrapper
from args import args

def make_env_specific_exp_name(env):
    if args.env == 'traffic':
        exp_name = '_'.join([item for item in [
            '{}'.format(args.policy),
            '{}'.format(args.env),
            'threshold={}'.format(args.proximity_threshold) 
                if args.policy in ['proximal_cg', 'adaptive'] else None,
            'edgePred={}'.format(args.edge_predictor_pred_method)
                if args.policy == 'adaptive' else None,
            'edgeInput={}'.format(args.edge_predictor_input_method)
                if args.policy == 'adaptive' else None,
            'difficulty={}'.format(args.difficulty),
            'add={}'.format(env.add_rate_max),
            'ent={}'.format(args.ent),
            'bs={}'.format(args.batch_size),
            'seed={}'.format(args.seed),
            'run',
        ] if item is not None])

    elif args.env == 'predprey':
        exp_name = '_'.join([item for item in [
            '{}'.format(args.policy),
            '{}'.format(args.env),
            'threshold={}'.format(args.proximity_threshold)
                if args.policy in ['proximal_cg', 'adaptive'] else None,
            'edgePred={}'.format(args.edge_predictor_pred_method)
                if args.policy == 'adaptive' else None,
            'edgeInput={}'.format(args.edge_predictor_input_method)
                if args.policy == 'adaptive' else None,
            'penalty={}'.format(args.penalty),
            'npred={}'.format(args.n_agents),
            'nprey={}'.format(args.n_entities),
            'grid={}'.format(args.n_grids),
            'avis={}'.format(args.agent_visible),
            'ent={}'.format(args.ent),
            'bs={}'.format(args.batch_size),
            'seed={}'.format(args.seed),
            'run',
        ] if item is not None])

    elif args.env == 'meet':
        exp_name = '_'.join([item for item in [
            '{}'.format(args.policy),
            '{}'.format(args.env),
            'threshold={}'.format(args.proximity_threshold)
                if args.policy in ['proximal_cg', 'adaptive'] else None,
            'edgePred={}'.format(args.edge_predictor_pred_method)
                if args.policy == 'adaptive' else None,
            'edgeInput={}'.format(args.edge_predictor_input_method)
                if args.policy == 'adaptive' else None,
            'n={}'.format(args.n_agents),
            'grid={}'.format(args.n_grids),
            'avis={}'.format(args.agent_visible),
            'ent={}'.format(args.ent),
            'bs={}'.format(args.batch_size),
            'seed={}'.format(args.seed),
            'run',
        ] if item is not None])

    else:
        NotImplementedError
    
    if args.eval:
        exp_name = 'eval_' + exp_name
    return exp_name


def make_env():
    if args.env == 'traffic':
        args.batch_size = 45000 #45000
        args.n_trajs_limit = None
        args.limit_by_traj = False
        args.ent = 0.05
        if not args.self_connected_adj:
            args.inverse_D = False
        args.proximity_threshold = 1.5
        env = TrafficJunctionWrapper(
            centralized=True, # CT
            difficulty=args.difficulty,
            self_connected_adj=args.self_connected_adj,
            inv_D=args.inverse_D,
            proximity_threshold=args.proximity_threshold)
        if args.policy == 'proximal_cg':
            try:
                args.proximity_threshold = float(
                    input('>>> [Press Enter for default value 1.5] proximity_threshold = '))
                print('>>> Setting proximity_threshold = {:.1f}'.format(args.proximity_threshold))
            except:
                print('>>> Using default value: proximity_threshold = 1.5')
                args.proximity_threshold = 1.5
    
    elif args.env == 'meet':
        args.batch_size = 45000
        args.n_trajs_limit = None
        args.limit_by_traj = False
        args.max_episode_steps = 100
        args.agent_visible = False
        args.n_agents = 10
        args.n_grids = 14
        args.ent = 0.35
        if not args.self_connected_adj:
            args.inverse_D = False
        if args.policy == 'proximal_cg':
            try:
                args.proximity_threshold = float(
                    input('>>> [Press Enter for default value 2.0] proximity_threshold = '))
                print('>>> Setting proximity_threshold = {:.1f}'.format(args.proximity_threshold))
            except:
                print('>>> Using default value: proximity_threshold = 2.0')
                args.proximity_threshold = 2.0
        env = MeetWrapper(
            centralized=True, # CT
            n_agents=args.n_agents,
            grid_size=args.n_grids,
            self_connected_adj=args.self_connected_adj,
            max_steps=args.max_episode_steps,
            inv_D=args.inverse_D,
            proximity_threshold=args.proximity_threshold,
            other_agent_visible=args.agent_visible,)

    elif args.env == 'predprey':
        args.max_episode_steps = 200
        args.n_agents = 8
        args.n_preys = 8
        args.n_grids = 10
        args.step_cost = -0.1
        args.limit_by_traj = True
        args.n_trajs_limit = 400
        #[Nominal value, not really used] DICG uses 80000:
        args.batch_size = args.n_trajs_limit * args.max_episode_steps
        args.ent = 0.1 # important
        if not args.self_connected_adj:
            args.inverse_D = False
        if args.policy == 'proximal_cg':
            try:
                args.proximity_threshold = float(
                    input('>>> [Press Enter for default value 2.0] proximity_threshold = '))
                print('>>> Setting proximity_threshold = {:.1f}'.format(args.proximity_threshold))
            except:
                print('>>> Using default value: proximity_threshold = 2.0')
                args.proximity_threshold = 2.0
        env = PredatorPreyWrapper(
            centralized=True, # CT
            other_agent_visible=args.agent_visible,
            proximity_threshold=args.proximity_threshold,
            inv_D=args.inverse_D,
            self_connected_adj=args.self_connected_adj,
            n_agents=args.n_agents,
            n_preys=args.n_preys,
            grid_shape=(args.n_grids, args.n_grids),
            max_steps=args.max_episode_steps,
            penalty=args.penalty,
            step_cost=args.step_cost,
            prey_capture_reward=10,
            render_adj=args.render_adj
        )

    else:
        NotImplementedError

    return make_env_specific_exp_name(env), env