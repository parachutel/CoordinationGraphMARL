import time
import dowel
from dowel import logger, tabular
from garage.misc.prog_bar_counter import ProgBarCounter
import numpy as np
import torch
from gym.wrappers import RecordVideo
from tqdm import tqdm

def record_gym_video(env, path, n_episodes=10, policy=None):
    recorder = RecordVideo(env, path)
    obses = recorder.reset()

    pbar = tqdm(range(n_episodes))
    for i_eps in pbar:
        pbar.set_postfix(episode=i_eps + 1)
        for _ in range(env.max_steps):

            # increment steps and episodes
            recorder.step_id += 1
            if recorder.recording:
                recorder.video_recorder.capture_frame()
                recorder.recorded_frames += 1

            if policy:
                adjs = env.get_proximity_adj_mat() if policy.proximity_adj else None
                # print(env.adj_raw)
                # env.render()
                # input()
                avail_actions = env.get_avail_actions()
                actions = policy.get_actions(obses, avail_actions, 
                                             adjs=adjs, greedy=True)[0]
            else:
                actions = [env.action_space.sample() for _ in range(env.n_agents)]

            obses, _, done, infos = env.step(actions)

            if done:
                for _ in range(5): # insert a pause
                    recorder.video_recorder.capture_frame()
                recorder.reset()
                break

    recorder.close_video_recorder()


def standard_eval(env, policy, n_episodes=20, greedy=True, load_from_file=False, 
                  render=False, recorder=None, max_steps=10000):
    if recorder is not None:
        render = False # force off
    if load_from_file:
        logger.add_output(dowel.StdOutput())
    logger.log('Evaluating policy, {} episodes, greedy = {} ...'.format(
        n_episodes, greedy))
    episode_rewards = []
    pbar = ProgBarCounter(n_episodes)
    for e in range(n_episodes):
        obs = env.reset()
        policy.reset([True])
        terminated = False
        t = 0
        episode_rewards.append(0)
        while not terminated:
            if render:
                env.render()
                # time.sleep(0.05)
            if recorder is not None:
                recorder.capture_frame()
            if not env.centralized:
                # obs.shape = (n_agents, n_envs, obs_dim)
                obs = torch.Tensor(obs).unsqueeze(1) # add n_envs dim
                avail_actions = torch.Tensor(env.get_avail_actions()).unsqueeze(1)
                actions, agent_infos = policy.get_actions(obs, 
                    avail_actions, greedy=greedy)
                if len(actions.shape) == 3: # n-d action
                    actions = actions[:, 0, :]
                elif len(actions.shape) == 2: # 1-d action
                    actions = actions[:, 0]
                obs, reward, terminated, info = env.step(actions) # n_env = 1
                terminated = all(terminated) 
            else:
                # obs.shape = (n_envs, n_agents * obs_dim)
                obs = np.array([obs])
                avail_actions = np.array([env.get_avail_actions()])
                actions, agent_infos = policy.get_actions(obs, 
                    avail_actions, greedy=greedy)
                obs, reward, terminated, info = env.step(actions[0]) # n_env = 1
            t += 1
            if t > max_steps:
                terminated = True
            episode_rewards[-1] += np.mean(reward)
        pbar.inc(1)
    pbar.stop()
    policy.reset([True])
    avg_return = np.mean(episode_rewards)
    logger.log('EvalAvgReturn: {}'.format(avg_return))
    if not load_from_file:
        tabular.record('EvalAvgReturn', avg_return)