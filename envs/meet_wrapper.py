import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../')

from envs.ma_gym.envs.meet_in_the_maze import Meet, PRE_IDS
import gym
import time
import torch
import copy
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from tqdm import tqdm

class MeetWrapper(Meet):

    def __init__(self, 
                 centralized, 
                 other_agent_visible=False, 
                 self_connected_adj=False,
                 inv_D=False,
                 proximity_threshold=4,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._agent_visible = other_agent_visible
        self.action_space = self.action_space[0]
        if self._agent_visible:
            mask_size = np.prod(self._agent_view_mask)
            self._obs_low = np.array([0., 0.] + [0.] * mask_size * 2 + [0.0], dtype=np.float32)
            self._obs_high = np.array([1., 1.] + [1.] * mask_size * 2 + [1.0], dtype=np.float32)
            self.observation_space = gym.spaces.Box(self._obs_low, self._obs_high)
        else:
            self.observation_space = self.observation_space[0]
        self.centralized = centralized
        if centralized:
            self.observation_space = gym.spaces.Box(
                low=np.array(list(self.observation_space.low) * self.n_agents),
                high=np.array(list(self.observation_space.high) * self.n_agents)
            )
        
        self.alive_mask = np.array([True] * self.n_agents)
        self.self_connected_adj = self_connected_adj
        self.inv_D = inv_D
        self.threshold = proximity_threshold
        self.pickleable = True
        self.metric_name = 'EvalAverageReturn'

    def get_avail_actions(self):
        avail_actions = [[1] * self.action_space.n for _ in range(self.n_agents)]
        if not self.centralized:
            return avail_actions
        else:
            return np.concatenate(avail_actions)

    def get_agent_obs(self):
        obs = super().get_agent_obs()
        if self._agent_visible:
            for i_agent in range(self.n_agents):
                pos = self.agent_pos[i_agent]
                # check if other agents are in the view area
                _agent_pos = np.zeros(self._agent_view_mask)
                for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                    for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                        if PRE_IDS['agent'] in self._full_obs[row][col]:
                            _agent_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1

                obs[i_agent].extend(_agent_pos.flatten().tolist())
        return obs

    def step(self, actions):
        obses, rewards, dones, infos = super().step(actions)
        if not self.centralized:
            return obses, rewards, dones, infos
        else:
            return np.concatenate(obses), np.mean(rewards), np.all(dones), infos

    def reset(self):
        obses = super().reset()
        self.get_proximity_adj_mat()
        if not self.centralized:
            return obses
        else:
            return np.concatenate(obses)
    
    def analyze_adjs(self, adjs):
        results = {}
        max_edged_dist = -1
        for i in range(self.n_agents - 1):
            for j in range(i + 1, self.n_agents):
                if adjs[i, j] == 1:
                    pi, pj = self.agent_pos[i], self.agent_pos[j]
                    dist = np.sqrt((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2)
                    max_edged_dist = max(max_edged_dist, dist)
        results['max_edged_dist'] = max_edged_dist
        return results

    def eval(self, epoch, policy, n_eval_episodes=100, greedy=True, 
            visualize=False, log=None, tbx=None, tabular=None):
        eval_avg_return = 0

        with torch.no_grad(), tqdm(total=n_eval_episodes) as progress_bar:
            for i_ep in range(n_eval_episodes):
                # Start episode
                obses = self.reset() # (n_agents, obs_dim)
    
                for t in range(self.max_steps):
                    if policy.proximity_adj:
                        adjs = self.get_proximity_adj_mat()
                        alive_masks = None
                    else:
                        adjs = None
                        alive_masks = np.array(self.alive_mask)
                    avail_actions = self.get_avail_actions()
                    # print('proximity adjs =', (torch.tensor(self.get_proximity_adj_mat()) > 0).float())
                    actions, agent_infos_n = policy.get_actions(
                        obses, avail_actions, adjs=adjs, 
                        alive_masks=alive_masks, greedy=greedy)
                    
                    if visualize:
                        self.render()
                        time.sleep(0.1)
                        # analysis_results = self.analyze_adjs(agent_infos_n['adjs'])
                        # print(analysis_results)
                        # input()

                    next_obses, rewards, done, info = self.step(actions)

                    eval_avg_return += np.mean(rewards)
                    if done:
                        break
                    obses = next_obses
                # end episode
                progress_bar.set_postfix(metric='{:.2f}'.format(
                    eval_avg_return / (i_ep + 1)))
                progress_bar.update(1)
            # end eval

        eval_avg_return /= n_eval_episodes

        log_strs = [
            'avg_return         {:.2f}'.format(eval_avg_return),
        ]

        tbx_results = {
            'avg_return': eval_avg_return,
        }

        # Logging...
        log.info('Eval' + '-' * 36)
        for item in log_strs:
            log.info(item)
        log.info('-' * 40)

        for k, v in tbx_results.items():
            if k[0] != ':':
                tbx.add_scalar(f'eval/{k}', v, epoch)
            else:
                tbx.add_histogram(f'eval/{k[1:]}', v, epoch)
        
        tabular.record(self.metric_name, eval_avg_return)

        return eval_avg_return # saver metric

    def get_proximity_adj_mat(self, raw=False):
        adj = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents - 1):
            for j in range(i + 1, self.n_agents):
                pi, pj = self.agent_pos[i], self.agent_pos[j]
                dist = np.sqrt((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2)
                if dist <= self.threshold:
                    adj[i, j] = 1
                    adj[j, i] = 1
        self.adj_raw = copy.deepcopy(adj)
        if raw:
            return adj
        if not self.inv_D:
            adj = (adj + np.eye(self.n_agents)) if self.self_connected_adj else adj
            sqrt_D = np.diag(np.sqrt(np.sum(adj, axis=1)))
            adj_renormalized = sqrt_D @ adj @ sqrt_D
        else:
            adj = adj + np.eye(self.n_agents)
            inv_sqrt_D = np.diag(np.sum(adj, axis=1) ** (-0.5))
            adj_renormalized = inv_sqrt_D @ adj @ inv_sqrt_D
        
        return adj_renormalized


if __name__ == '__main__':
    env = MeetWrapper(centralized=True, 
                      grid_size=14, 
                      n_agents=10,
                      max_steps=100)
    print(env.action_space)
    print(env.observation_space)
    obs = env.reset()
    print(obs)
    from utils import record_gym_video
    record_gym_video(env, path='./meet_recording.test')

    ## for live render
    # for t in range(env.max_steps):
    #     actions = [env.action_space.sample() for _ in range(env.n_agents)]
    #     env.render()
    #     # input()
    #     time.sleep(0.1)
    #     obses, rewards, done, info = env.step(actions)
    #     # print(obses, rewards, done, info)

    
