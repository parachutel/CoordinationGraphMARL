import os
import sys
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_FILE_PATH)
sys.path.append(CURRENT_FILE_PATH + '/../')

import copy
import logging
import random

import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text, draw_line

logger = logging.getLogger(__name__)


class Meet(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 20}

    def __init__(self, 
                 grid_size=14, 
                 n_agents=10,
                 full_observable=False, 
                 step_cost=-0.01, 
                 max_steps=100,
                 render_adj=False,):

        self.n_grids = grid_size
        self._grid_shape = (grid_size, grid_size)
        self.n_agents = n_agents
        self.max_steps = max_steps
        self._step_count = None
        self._step_cost = step_cost
        self._agent_view_mask = (5, 5)
        self._agent_view_width = (self._agent_view_mask[0] - 1) // 2

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        
        with open(CURRENT_FILE_PATH + '/map_{}.txt'.format(grid_size)) as f:
            lines = f.read().splitlines()
            self.map_config = [line.split(' ') for line in lines]
        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]
        self.viewer = None
        self.full_observable = full_observable

        # agent pos (2), view mask (25), step (1)
        mask_size = np.prod(self._agent_view_mask)
        self._obs_high = np.array([1., 1.] + [1.] * mask_size + [1.0], dtype=np.float32)
        self._obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0], dtype=np.float32)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._total_episode_reward = None

        self.adj_raw = None
        self.render_adj = render_adj

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_grid(self):
        _grid = [[
            PRE_IDS['empty'] if self.map_config[row][col] == '0' else PRE_IDS['wall']
            for col in range(self._grid_shape[1])
        ] for row in range(self._grid_shape[0])]
        return _grid

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        for agent_i in range(self.n_agents):
            while True:
                pos = [random.randint(0, self._grid_shape[0] - 1), random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos):
                    self.agent_pos[agent_i] = pos
                    break
            self.__update_agent_view(agent_i)

        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            # check if wall is in the view area
            _wall_pos = np.zeros(self._agent_view_mask)
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['wall'] in self._full_obs[row][col]:
                        _wall_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = -1  # get relative position for the wall loc.

            _agent_i_obs += _wall_pos.flatten().tolist()  # adding wall pos in observable area
            _agent_i_obs += [self._step_count / self.max_steps]  # adding time
            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}

        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]

        return self.get_agent_obs()

    def __wall_exists(self, pos):
        row, col = pos
        return PRE_IDS['wall'] in self._base_grid[row][col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        return next_pos

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def _neighbour_agents(self, pos):
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self._full_obs[x][y].split(PRE_IDS['agent'])[1]) - 1)
        return _count, agent_id

    def step(self, agents_action):
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)
            for agent_j in range(self.n_agents):
                if agent_i != agent_j:
                    rewards[agent_i] -= np.sqrt(
                        (self.agent_pos[agent_i][0] - self.agent_pos[agent_j][0]) ** 2
                        + (self.agent_pos[agent_i][1] - self.agent_pos[agent_j][1]) ** 2) / 100

        if (self._step_count >= self.max_steps):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        return self.get_agent_obs(), rewards, self._agent_dones, {}

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):
                if self.__wall_exists((row, col)):
                    fill_cell(img, [row, col], cell_size=CELL_SIZE, fill=WALL_COLOR, margin=0.0)

        for agent_i in range(self.n_agents):
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)
        
        if self.adj_raw is not None and self.render_adj:
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if i != j and self.adj_raw[i, j] == 1:
                        draw_line(img, self.agent_pos[i], self.agent_pos[j], 
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
        

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)

CELL_SIZE = 35

WALL_COLOR = 'gray'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0'
}


if __name__ == '__main__':
    import time
    env = Meet()
    obses = env.reset()
    for _ in range(1000):
        actions = env.action_space_sample()
        obses, rewards, dones, info = env.step(actions)
        print(rewards)
        env.render()
        time.sleep(0.1)
