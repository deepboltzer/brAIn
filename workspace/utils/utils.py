from email import header
from math import factorial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


def record_video(env_id, model, video_length=500, prefix='', video_folder='./out'):
  """
  Records a video of the agent performing in set environment.
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """

  eval_env = DummyVecEnv([lambda: gym.make(env_id)])

  # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
  eval_env.close()


def load_log(file_path):
  """
  Loads data from an evaluations.npz file and returns it as pandas DataFrame.
  :param file_path: (str)
  """
  with np.load(file_path) as f:
    f = dict(f)
    data = {}
    data['timesteps'] = f['timesteps']
    data['mean_rew'] = [np.mean(x) for x in f['results']]
    data['mean_ep_length'] = [np.mean(x) for x in f['ep_lengths']]
    df = pd.DataFrame(data).set_index('timesteps')

    return df


def plot_log(data, title, xlabel='timesteps', ylabel='mean reward', figsize=(9,5), axis=[0, 500000, -100, 750]):
  """
  Takes pandas DataFrame and plots it.
  :param data: (DataFrame)
  :param title: (str)
  :param xlabel: (str)
  :param ylabel: (str)
  :param figsize: (tupel)
  :param axis: (list)
  """
  plt.figure(figsize=figsize)
  plt.plot(data.index, data['mean_rew'], color='darkslategray', alpha=1, zorder=0)
  plt.vlines(data.index, ymin=0, ymax=data['mean_rew'], color=np.where(data['mean_rew'] < 0, 'maroon', 'darkgreen'), alpha=1, zorder=1)
  plt.scatter(data.index, data['mean_rew'], color=np.where(data['mean_rew'] < 0, 'maroon', 'darkgreen'), marker='D', alpha=1, zorder=2)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.axis(axis) # plt.axis([xmin, xmax, ymin, ymax])
  plt.show()