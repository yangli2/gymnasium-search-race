r"""Launcher script for training Gym Search Race.

Example usage:

To run sweeps locally:
  python launcher.py --total_timesteps=100
"""

import json
import os
import subprocess
import sys

from absl import app
from absl import flags

_EXP_TITLE = flags.DEFINE_string(
    'exp_title', 'gym_search_race_train', 'Title of the experiment.'
)
flags.DEFINE_integer('total_timesteps', 10000000, 'Total timesteps to train')
flags.DEFINE_integer('max_parallel', 4, 'Maximum number of parallel runs')

FLAGS = flags.FLAGS

def main(_):
  workdir = f'/tmp/gym_search_race/runs'

  # 10 Different Hyperparameter Configurations for tuning
  sweeps_path = os.path.join(os.path.dirname(__file__), 'sweeps.json')
  if os.path.exists(sweeps_path):
    print(f'Loading dynamic sweeps from {sweeps_path}...')
    with open(sweeps_path, 'r') as f:
      hp_sweep = json.load(f)
  else:
    hp_sweep = [
        # New Baseline derived from previous sweep
        {
            'name': 'New_Baseline',
            'pi_layers': '64,64,64',
            'vf_layers': '256,256,256',
            'learning_rate': 5e-5,
            'gamma': 0.995,
            'n_steps': 4096,
            'clip_range': 0.2,
        },
        # Clipping lower
        {
            'name': 'Clip_0_1',
            'pi_layers': '64,64,64',
            'vf_layers': '256,256,256',
            'learning_rate': 5e-5,
            'gamma': 0.995,
            'n_steps': 4096,
            'clip_range': 0.1,
        },
        # Clipping slightly higher
        {
            'name': 'Clip_0_3',
            'pi_layers': '64,64,64',
            'vf_layers': '256,256,256',
            'learning_rate': 5e-5,
            'gamma': 0.995,
            'n_steps': 4096,
            'clip_range': 0.3,
        },
        # Clipping higher
        {
            'name': 'Clip_0_4',
            'pi_layers': '64,64,64',
            'vf_layers': '256,256,256',
            'learning_rate': 5e-5,
            'gamma': 0.995,
            'n_steps': 4096,
            'clip_range': 0.4,
        },
        # Even higher learning rate
        {
            'name': 'Learning_Rate_1e_4',
            'pi_layers': '64,64,64',
            'vf_layers': '256,256,256',
            'learning_rate': 1e-4,
            'gamma': 0.995,
            'n_steps': 4096,
            'clip_range': 0.2,
        },
        # Higher LR with higher clipping
        {
            'name': 'LR_1e_4_Clip_0_3',
            'pi_layers': '64,64,64',
            'vf_layers': '256,256,256',
            'learning_rate': 1e-4,
            'gamma': 0.995,
            'n_steps': 4096,
            'clip_range': 0.3,
        },
        # Higher LR with even higher clipping
        {
            'name': 'LR_1e_4_Clip_0_4',
            'pi_layers': '64,64,64',
            'vf_layers': '256,256,256',
            'learning_rate': 1e-4,
            'gamma': 0.995,
            'n_steps': 4096,
            'clip_range': 0.4,
        },
        # Very high learning rate
        {
            'name': 'Learning_Rate_3e_4',
            'pi_layers': '64,64,64',
            'vf_layers': '256,256,256',
            'learning_rate': 3e-4,
            'gamma': 0.995,
            'n_steps': 4096,
            'clip_range': 0.2,
        },
        # Very high learning rate with high clipping
        {
            'name': 'LR_3e_4_Clip_0_4',
            'pi_layers': '64,64,64',
            'vf_layers': '256,256,256',
            'learning_rate': 3e-4,
            'gamma': 0.995,
            'n_steps': 4096,
            'clip_range': 0.4,
        },
        # Moderate net
        {
            'name': 'Moderate_Net_LR_5e_5',
            'pi_layers': '64,64,64',
            'vf_layers': '128,128,128',
            'learning_rate': 5e-5,
            'gamma': 0.995,
            'n_steps': 4096,
            'clip_range': 0.2,
        },
    ]

  print(f"Launching sweeps locally. Logs and models will be saved to {workdir}")

  processes = []
  for hp in hp_sweep:
    name = hp['name']
    hp_args = hp.copy()
    del hp_args['name']

    cmd = [
        sys.executable,
        'train.py',
        f'--workdir={workdir}/{name}',
        f'--total_timesteps={FLAGS.total_timesteps}',
    ]
    for k, v in hp_args.items():
      cmd.append(f'--{k}={v}')

    print(f"Launching run: {name} with command: {' '.join(cmd)}")
    p = subprocess.Popen(cmd)
    processes.append(p)

    # limit concurrency
    if len(processes) >= FLAGS.max_parallel:
      for p in processes:
        p.wait()
      processes = []

  # Wait for any remaining processes
  for p in processes:
    p.wait()

if __name__ == '__main__':
  app.run(main)
