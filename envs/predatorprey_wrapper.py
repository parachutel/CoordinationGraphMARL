import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../')

from envs.ma_gym.envs.predator_prey import PredatorPrey
import gym
import time
import torch
import copy
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from tqdm import tqdm

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0'
}

class PredatorPreyWrapper(PredatorPrey):

    def __init__(self, 
                 centralized, 
                 other_agent_visible=False, 
                 self_connected_adj=False,
                 inv_D=False,
                 proximity_threshold=2,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.n_grids = self._grid_shape[0]
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
        self.threshold = proximity_threshold
        self.self_connected_adj = self_connected_adj
        self.inv_D = inv_D
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
                            # get relative position for the prey loc:
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

    def eval(self, epoch, policy, n_eval_episodes=100, greedy=True, 
            visualize=False, log=None, tbx=None, tabular=None):
        eval_avg_return = 0
        eval_env_steps = 0

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
                        # input()

                    next_obses, rewards, done, info = self.step(actions)

                    eval_avg_return += np.mean(rewards)
                    if done:
                        eval_env_steps += (t + 1)
                        break
                    obses = next_obses
                # end episode
                progress_bar.set_postfix(metric='{:.2f}'.format(
                    eval_avg_return / (i_ep + 1)))
                progress_bar.update(1)
            # end eval

        eval_avg_return /= n_eval_episodes
        eval_env_steps /= n_eval_episodes

        log_strs = [
            'avg_return         {:.2f}'.format(eval_avg_return),
            'avg_episode_steps  {}'.format(eval_env_steps)
        ]

        tbx_results = {
            'avg_return': eval_avg_return,
            'avg_episode_steps': eval_env_steps,
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
        
        tabular.record(self.metric_name, eval_env_steps)

        return eval_avg_return # saver metric

    def my_render(self, attention_weights=None, cell_size=35):
        """
            [deprecated]
        """
        if attention_weights is not None:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = Image.fromarray(self.render(mode='rgb_array'))
            
            # Plot attention weights for agent 0

            start_col, start_row = self.agent_pos[0]
            start_x, start_y = (start_row + 0.5) * cell_size, (start_col + 0.5) * cell_size
            for i in range(self.n_agents):
                if i == 0:
                    draw_circle(
                        img, self.agent_pos[i], cell_size=cell_size, 
                        fill=None, outline='green', radius=0.1,
                        width=int(20 * attention_weights[i]))
                else:
                    if attention_weights[i] == 0:
                        fill = None
                    else:
                        fill = 'green'
                    end_col, end_row = self.agent_pos[i]
                    end_x, end_y = (end_row + 0.5) * cell_size, (end_col + 0.5) * cell_size
                    ImageDraw.Draw(img).line(((start_x, start_y), (end_x, end_y)), 
                        fill=fill, width=int(20 * attention_weights[i]))

            img = np.asarray(img)
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            self.render(mode='human')

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

def draw_circle(image, pos, cell_size=50, fill='white', outline='black', 
                radius=0.3, width=1):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    gap = cell_size * radius
    x, y = row + gap, col + gap
    x_dash, y_dash = row + cell_size - gap, col + cell_size - gap
    ImageDraw.Draw(image).ellipse([(x, y), (x_dash, y_dash)], fill=fill, 
        outline=outline, width=width)


if __name__ == '__main__':
    env = PredatorPreyWrapper(centralized=True, 
                              grid_shape=(10, 10), 
                              n_agents=8, 
                              n_preys=8)
    print(env.action_space)
    print(env.observation_space)
    obs = env.reset()
    print(obs)
    from utils import record_gym_video
    record_gym_video(env, path='./predatorprey_recording.test')
    
    ## for live render
    #for t in range(100):
    #    actions = [env.action_space.sample() for _ in range(env.n_agents)]
    #    env.render()
    #    time.sleep(0.1)
    #    obses, rewards, done, info = env.step(actions)
    #    print(obses, rewards, done, info)
