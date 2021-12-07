import time
import numpy as np
from dowel import logger, tabular
from garage.misc.prog_bar_counter import ProgBarCounter
from garage.sampler.batch_sampler import BatchSampler



class CentralizedMAOnPolicySampler(BatchSampler):
    """
    Args:
        algo (garage.np.algos.RLAlgorithm): An algorithm instance.
        env (garage.envs.GarageEnv): An environement instance.
    """

    def __init__(self, algo, env, 
                 batch_size=60000, 
                 n_trajs_limit=None,
                 limit_by_traj=False):
        super().__init__(algo, env)
        self._env = env
        self.tot_num_env_steps = 0
        self.limit_by_traj = limit_by_traj
        self.batch_size = batch_size
        self.n_trajs_limit = n_trajs_limit
        self.n_samples = 0
        self.n_trajs = 0

        if limit_by_traj:
            assert n_trajs_limit is not None
        else:
            assert batch_size is not None
    
    def reset_sample_counters(self):
        self.n_samples = 0
        self.n_trajs = 0
    
    def loop_flag(self):
        return self.n_trajs < self.n_trajs_limit if self.limit_by_traj \
            else self.n_samples < self.batch_size

    def obtain_samples(self, itr, tbx):
        """
            batch_size is the number of env_steps
        """

        logger.log('Obtaining samples for iteration %d...' % itr)

        paths = []
        self.reset_sample_counters()
        if hasattr(self.env, 'curriculum_learning'):
            obses = self._env.reset(itr)
        else:
            obses = self._env.reset()

        running_path = None

        pbar = ProgBarCounter(self.n_trajs_limit if self.limit_by_traj 
                              else self.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy
        
        while self.loop_flag():
            t = time.time()
            policy.reset(True)

            avail_actions = self._env.get_avail_actions()

            if self.algo.policy.proximity_adj:
                adj = self._env.get_proximity_adj_mat() # renormalized adj
                alive_mask = None
            else:
                adj = None # will be computed by policy
                alive_mask = np.array(self._env.alive_mask) 
                # most primitive form, (n_agents, )
                
            actions, _ = policy.get_actions(obses, avail_actions, 
                adjs=adj, alive_masks=alive_mask)

            adj = np.asarray(adj)
            adj = np.reshape(adj, (-1,)) # flatten like obses

            policy_time += time.time() - t
            t = time.time()
            state = obses
            next_obses, rewards, dones, env_infos = self._env.step(actions)
            env_time += time.time() - t
            t = time.time()
            
            if running_path is None:
                running_path = dict(observations=[],
                                    states=[],
                                    actions=[],
                                    avail_actions=[],
                                    alive_masks=[],
                                    rewards=[],
                                    adjs=[],
                                    dones=[])
            running_path['observations'].append(obses)
            running_path['states'].append(state)
            running_path['actions'].append(actions)
            running_path['avail_actions'].append(avail_actions)
            running_path['alive_masks'].append(alive_mask)
            running_path['rewards'].append(rewards)
            running_path['dones'].append(dones)
            running_path['adjs'].append(adj)
    
            obses = next_obses

            if dones:
                paths.append(
                    dict(observations=np.asarray(running_path['observations']),
                         states=np.asarray(running_path['states']),
                         actions=np.asarray(running_path['actions']),
                         avail_actions=np.asanyarray(running_path['avail_actions']),
                         alive_masks=np.asanyarray(running_path['alive_masks']),
                         rewards=np.asarray(running_path['rewards']),
                         dones=np.asarray(running_path['dones']),
                         adjs=np.asarray(running_path['adjs']),
                    )
                )
                self.n_samples += len(running_path['rewards'])
                self.n_trajs += 1
                self.tot_num_env_steps += len(running_path['rewards'])
                running_path = None
                obses = self._env.reset()
                if self.limit_by_traj:
                    pbar.inc(1)

            process_time += time.time() - t
            if not self.limit_by_traj:
                pbar.inc(1)
            # print(self.n_samples)

        pbar.stop()

        tabular.record('PolicyExecTime', policy_time)
        tabular.record('EnvExecTime', env_time)
        tabular.record('ProcessExecTime', process_time)
        tabular.record('EnvSteps', self.tot_num_env_steps)
        tabular.record('BatchSize', self.n_samples)
        tabular.record('NumTrajs', self.n_trajs)

        tbx.add_scalar('train/env_steps', self.tot_num_env_steps, itr)
        tbx.add_scalar('train/batch_size', self.n_samples, itr)
        tbx.add_scalar('train/n_trajs', self.n_trajs, itr)
        
        return paths
