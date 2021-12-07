# Using local gym
import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../')


from envs.traffic_junction.traffic_junction import TrafficJunctionEnv
from PIL import ImageColor
from envs.ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
import gym
from gym.utils import seeding
import numpy as np

from envs.utils import standard_eval
import time
import copy
import torch
from tqdm import tqdm

class TrafficJunctionWrapper(TrafficJunctionEnv):

    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 20}

    def __init__(self, 
                 centralized,
                 vision=0,
                 difficulty='hard',
                 self_connected_adj=False,
                 inv_D=False,
                 proximity_threshold=1.5,
                 render_adj=False):

        super().__init__(vision=vision, difficulty=difficulty)

        self.t = 0
        self.n_agents = self.ncar

        self.curriculum_learning = False

        self.centralized = centralized
        self.original_obs_space = copy.deepcopy(self.observation_space)

        if centralized:
            self.observation_space = gym.spaces.Box(
                low=np.array(self.obs_dim * [-1] * self.n_agents, dtype=np.float32),
                high=np.array(self.obs_dim * [1] * self.n_agents, dtype=np.float32)
            )
        else:
            self.observation_space = self.observation_space[0]
        self.action_space = self.action_space[0]

        self.self_connected_adj = self_connected_adj
        self.inv_D = inv_D
        self.threshold = proximity_threshold
        self.pickleable = True
        self.metric_name = 'SuccessRate'

        self.viewer = None
        self._base_img = draw_grid(self.dim, self.dim, cell_size=CELL_SIZE, fill='white', line_color='white')

        self.adj_raw = None
        self.render_adj = render_adj

    def get_avail_actions(self):
        avail_actions = [[1] * self.action_space.n for _ in range(self.n_agents)]
        if not self.centralized:
            return avail_actions
        else:
            return np.concatenate(avail_actions)

    def step(self, actions):
        obses, rewards, dones, infos = super().step(actions)
        if self.t >= self.max_steps:
            dones = True
        self.t += 1
        if not self.centralized:
            return obses, rewards, [bool(dones)] * self.n_agents, infos
        else:
            return np.concatenate(obses), np.mean(rewards), np.all(dones), infos

    def reset(self, epoch=None):
        self.t = 0
        obses = super().reset(epoch)
        self.get_proximity_adj_mat()
        if not self.centralized:
            return obses
        else:
            return np.concatenate(obses)

    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        grid = self.grid.copy().astype(object)

        for row in range(self.dim):
            for col in range(self.dim):
                if grid[row][col] != self.OUTSIDE_CLASS: # colorize road
                    fill_cell(img, [row, col], cell_size=CELL_SIZE, fill=ROAD_COLOR, margin=0)

        grid[grid != self.OUTSIDE_CLASS] = '_'
        grid[grid == self.OUTSIDE_CLASS] = ''

        for i_agent in range(self.n_agents):
            row, col = self.car_loc[i_agent][0], self.car_loc[i_agent][1]
            if row == col == 0:
                continue
            
            if self.car_last_act[i_agent] == 0: # GAS
                if grid[row][col] != 0:
                    grid[row][col] = str(grid[row][col]).replace('_','') + '<>'
                else:
                    grid[row][col] = '<>'
            else: # BRAKE
                if grid[row][col] != 0:
                    grid[row][col] = str(grid[row][col]).replace('_','') + '<b>'
                else:
                    grid[row][col] = '<b>'

            item = grid[row][col]

            if item != '_':
                if '<>' in item and len(item) > 3: #CRASH, one car accelerates
                    fill = CRASH_COLOR
                elif '<>' in item: #GAS
                    fill = GAS_COLOR
                elif 'b' in item and len(item) > 3: #CRASH
                    fill = CRASH_COLOR
                elif 'b' in item:
                    fill = BREAK_COLOR
                
                fill_cell(img, (row, col), cell_size=CELL_SIZE, fill=fill, margin=0.1)
                write_cell_text(img, text=str(i_agent + 1), pos=(row, col), cell_size=CELL_SIZE,
                                fill='white', margin=0.4)
        
        if self.adj_raw is not None and self.render_adj:
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if i != j and self.adj_raw[i, j] == 1:
                        draw_line(img, self.car_loc[i], self.car_loc[j], 
                                  cell_size=CELL_SIZE, line_color='red')

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def eval(self, epoch, policy, n_eval_episodes=100, greedy=True, 
            visualize=False, log=None, tbx=None, tabular=None, aux=None):
        eval_avg_return = 0
        eval_success_rate = 0
        if policy.proximity_adj:
            eval_edge_pred_acc = []
        action_distribution = []

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
                    actions, agent_infos_n = policy.get_actions(
                        obses, avail_actions, adjs=adjs, 
                        alive_masks=alive_masks, greedy=greedy)
                    
                    if visualize:
                        self.render()
                        time.sleep(0.1)

                    next_obses, rewards, done, info = self.step(actions)
                
                    eval_avg_return += np.mean(rewards)
                    action_distribution += list(actions)
                    if done:
                        break
                    obses = next_obses
                # end episode
                eval_success_rate += self.stat['success']

                progress_bar.set_postfix(metric='{:.2f}'.format(
                    eval_success_rate / (i_ep + 1)))
                progress_bar.update(1)
            # end eval

        eval_avg_return /= n_eval_episodes
        eval_success_rate /= n_eval_episodes

        log_strs = [
            'add_rate      {}'.format(self.stat['add_rate']),
            'avg_return    {:.4f}'.format(eval_avg_return),
            'success_rate  {:.2f}%'.format(eval_success_rate * 100)
        ]

        tbx_results = {
            'avg_return': eval_avg_return,
            'success_rate': eval_success_rate,
            ':action_distribution': np.array(action_distribution)
        }

        # if policy.proximity_adj:
        #     log_strs.append('edge_pred_acc {:.2f}%'.format(np.mean(eval_edge_pred_acc) * 100))
        #     tbx_results['edge_pred_acc'] = np.mean(eval_edge_pred_acc)

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
        
        tabular.record(self.metric_name, eval_success_rate)

        return eval_success_rate # saver metric
    
    def get_proximity_adj_mat(self, raw=False):
        adj = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents - 1):
            for j in range(i + 1, self.n_agents):
                if self.alive_mask[i] > 0 and self.alive_mask[j] > 0:
                    pi, pj = self.car_loc[i], self.car_loc[j]
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
    

    def edge_prediction_data_collection(self, max_n_positive_samples=3e6,
                                        pos_neg_ratio=2, self_connections=False):
        '''
            [deprecated]
        '''
        positive_inputs = []
        positive_labels = []
        negative_inputs = []
        negative_labels = []
        n_positive_samples = 0
        n_env_steps = 0
        max_steps = self.max_steps # cache
        add_rate_max = self.add_rate_max # cache
        # alter params
        self.max_steps = np.inf
        self.add_rate_max = 0.05
        diag_offset = 0 if self_connections else 1

        obses = self.reset()
        with tqdm(total=int(max_n_positive_samples)) as progress_bar:
            while True:
                adj_renormalized = self.get_proximity_adj_mat()
                adj_raw = np.array(adj_renormalized > 0, dtype=np.float32)
                adj_triu = np.triu(adj_raw, k=diag_offset)

                actions = [self.action_space.sample() for _ in range(self.n_agents)]
                obses, _, _, _ = self.step(actions)
                obses = np.reshape(obses, (self.n_agents, -1)).astype(np.float32)

                positive_sample_coords = np.where(adj_triu == 1)
                negative_sample_coords = np.where(adj_triu == 0)
                alive_agent_idxs = np.where(self.alive_mask)[0]
                
                positive_samples_count = len(positive_sample_coords[0])
                negative_samples_count = 0

                if positive_samples_count == 0:
                    # No positive sample available
                    continue
                
                # Collect positive samples
                for i, j in zip(*positive_sample_coords):
                    positive_inputs.append(obses[i] + obses[j])
                    positive_labels.append(1.)
                
                # Collect negative samples
                while True:
                    # First consider alive agent pairs
                    for i, j in zip(*negative_sample_coords):
                        if i < j and i in alive_agent_idxs and j in alive_agent_idxs:
                            negative_samples_count += 1
                            negative_inputs.append(obses[i] + obses[j])
                            negative_labels.append(0.)
                            if negative_samples_count >= int(pos_neg_ratio * positive_samples_count):
                                break
                    
                    if negative_samples_count >= int(pos_neg_ratio * positive_samples_count):
                        break

                    # Then consider one alive agent in an agent pairs
                    for i, j in zip(*negative_sample_coords):
                        if i < j and (
                            (i in alive_agent_idxs and j not in alive_agent_idxs) or
                            (i not in alive_agent_idxs and j in alive_agent_idxs)):
                            negative_samples_count += 1
                            negative_inputs.append(obses[i] + obses[j])
                            negative_labels.append(0.)
                            if negative_samples_count >= int(pos_neg_ratio * positive_samples_count):
                                break

                    if negative_samples_count >= int(pos_neg_ratio * positive_samples_count):
                        break
                    
                    # Finally consider both dead agent pairs
                    for i, j in zip(*negative_sample_coords):
                        if i < j and i not in alive_agent_idxs and j not in alive_agent_idxs:
                            negative_samples_count += 1
                            negative_inputs.append(obses[i] + obses[j])
                            negative_labels.append(0.)
                            if negative_samples_count >= int(pos_neg_ratio * positive_samples_count):
                                break

                    if negative_samples_count >= int(pos_neg_ratio * positive_samples_count):
                        break

                n_positive_samples += positive_samples_count
                n_env_steps += 1
                progress_bar.update(positive_samples_count)
                progress_bar.set_postfix(
                    n_neg='{}'.format(len(negative_labels)),
                    n_pos='{}'.format(len(positive_labels))
                )

                if n_positive_samples >= max_n_positive_samples:
                    break
            # end while
        
        self.max_steps = max_steps # recover from cache
        self.add_rate_max = add_rate_max

        inputs = np.array(negative_inputs + positive_inputs, dtype=np.float32)
        labels = np.array(negative_labels + positive_labels, dtype=np.float32)
        random_idxs = np.random.permutation(len(labels))
        inputs = inputs[random_idxs]
        labels = labels[random_idxs]
        
        np.savez('./traffic_edge_prediction_data.npz',
                 inputs=inputs,
                 labels=labels,
                 n_agents=self.n_agents,
                 difficulty=self.difficulty)

CELL_SIZE = 35

ROAD_COLOR = 'gray'
CRASH_COLOR = 'yellow'
GAS_COLOR = 'red'
BREAK_COLOR = 'cyan'

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0'
}

if __name__ == '__main__':
    env = TrafficJunctionWrapper(
        centralized=True, # CT
        difficulty='hard',
        self_connected_adj=False,
        inv_D=False,
        proximity_threshold=1.5)
    
    # env.edge_prediction_data_collection()

    print(env.action_space)
    print(env.observation_space)
    obs = env.reset()
    print(obs)
    from utils import record_gym_video
    record_gym_video(env, path='./traffic_junction_recording.test')
    
    ## for live render
    #import time
    #for _ in range(env.max_steps):
    #    env.render()
    #    time.sleep(0.5)
    #    actions = [env.action_space.sample() for _ in range(env.n_agents)]
    #    env.step(actions)

    
