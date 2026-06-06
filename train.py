r"""Train a PPO agent on the SearchRace environment.

Example usage to run locally for testing:
  blaze run //experimental/users/yangliyl/gym_search_race:train -- \\
      --total_timesteps=100
"""

import os

from absl import app
from absl import flags
from torch import nn

# Import to register the environments in gym
import gymnasium_search_race  # pylint: disable=unused-import

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env


FLAGS = flags.FLAGS
flags.DEFINE_integer("total_timesteps", 10000000, "Total timesteps to train")
flags.DEFINE_string(
    "env_id", "gymnasium_search_race/MadPodRacingDiscrete-v2", "Environment ID"
)
flags.DEFINE_string(
    "workdir", "/tmp/gym_search_race", "Working directory for logs and models"
)
flags.DEFINE_integer("n_envs", 32, "Number of parallel environments")

# --- Tunable Hyperparameters ---
# Model Architecture
flags.DEFINE_list("pi_layers", ["64", "64", "64"], "Sizes of policy net layers")
flags.DEFINE_list(
    "vf_layers", ["128", "128", "128"], "Sizes of value net layers"
)

# RL Hyperparameters
flags.DEFINE_integer(
    "n_steps", 4096, "Number of steps to run for each environment per update"
)
flags.DEFINE_integer("batch_size", 64, "Minibatch size")
flags.DEFINE_integer(
    "n_epochs", 10, "Number of epochs when optimizing the surrogate loss"
)
flags.DEFINE_float("gamma", 0.99, "Discount factor")
flags.DEFINE_float("learning_rate", 1e-5, "Learning rate")
flags.DEFINE_float("clip_range", 0.2, "Clipping parameter")
flags.DEFINE_float(
    "ent_coef", 0.01, "Entropy coefficient for the loss calculation"
)
flags.DEFINE_float(
    "vf_coef", 0.5, "Value function coefficient for the loss calculation"
)
flags.DEFINE_string(
    "activation_fn", "ReLU", "Activation function: ReLU, Tanh, ELU"
)
flags.DEFINE_float(
    "log_std_init",
    -2.0,
    "Initial log standard deviation for continuous actions",
)
flags.DEFINE_bool(
    "use_vec_normalize",
    False,
    "Use VecNormalize wrapper for obs and rewards",
)
flags.DEFINE_bool(
    "ortho_init", False, "Use orthogonal initialization for weights"
)
flags.DEFINE_float("gae_lambda", 0.95, "GAE parameter lambda")
flags.DEFINE_float(
    "clip_range_vf", -1.0, "Value function clipping. < 0 means None"
)
flags.DEFINE_string(
    "thrust_mapping",
    "linear",
    "Thrust mapping: linear, power_0_2, power_0_5, cubic, threshold",
)
flags.DEFINE_integer("seed", None, "Random seed for training reproducibility")
flags.DEFINE_bool(
    "pruned_actions", False, "Prune discrete steering action space"
)
flags.DEFINE_bool(
    "smart_thrust_cap", False, "Cap thrust to 0 if heading error > 90 degrees"
)
flags.DEFINE_float(
    "align_reward_beta", 0.0, "Beta for directional velocity alignment reward"
)
flags.DEFINE_bool(
    "use_action_history", False, "Include previous action in observations"
)
flags.DEFINE_float(
    "smoothness_weight", 0.0, "Weight for steering smoothness penalty"
)
flags.DEFINE_float(
    "potential_alpha", 0.0, "Alpha for potential-based progress reward"
)
flags.DEFINE_integer(
    "non_uniform_grid", 0, "Non-uniform grid size (7 or 9, 0 for default)"
)
flags.DEFINE_integer(
    "action_history_len", 0, "Number of steps of action history in observations"
)
flags.DEFINE_bool(
    "use_asymmetric_history",
    False,
    "Include only previous steering actions in observations",
)
flags.DEFINE_bool(
    "use_damped_thrust",
    False,
    "Include damped thrust memory in observations",
)
flags.DEFINE_float(
    "damped_thrust_beta",
    0.9,
    "Decay rate for damped thrust memory",
)
flags.DEFINE_integer(
    "thrust_levels",
    2,
    "Number of discrete thrust levels (2 or 3)",
)
flags.DEFINE_integer(
    "custom_mid_thrust",
    None,
    "Custom middle thrust value for 3-level thrust",
)


def main(argv):
  del argv
  if not os.path.exists(FLAGS.workdir):
    os.makedirs(FLAGS.workdir, exist_ok=True)

  print(
      f"Creating environment {FLAGS.env_id} with {FLAGS.n_envs} parallel"
      " instances..."
  )
  env_kwargs = dict(
      laps=3,
      car_max_thrust=200,
      sequential_maps=True,
      thrust_mapping=FLAGS.thrust_mapping,
      pruned_actions=FLAGS.pruned_actions,
      smart_thrust_cap=FLAGS.smart_thrust_cap,
      align_reward_beta=FLAGS.align_reward_beta,
      use_action_history=FLAGS.use_action_history,
      smoothness_weight=FLAGS.smoothness_weight,
      potential_alpha=FLAGS.potential_alpha,
      non_uniform_grid=FLAGS.non_uniform_grid,
      action_history_len=FLAGS.action_history_len,
      use_asymmetric_history=FLAGS.use_asymmetric_history,
      use_damped_thrust=FLAGS.use_damped_thrust,
      damped_thrust_beta=FLAGS.damped_thrust_beta,
      thrust_levels=FLAGS.thrust_levels,
      custom_mid_thrust=FLAGS.custom_mid_thrust,
  )

  env = make_vec_env(
      FLAGS.env_id,
      n_envs=FLAGS.n_envs,
      env_kwargs=env_kwargs,
      seed=FLAGS.seed,
  )

  if FLAGS.use_vec_normalize:
    from stable_baselines3.common.vec_env.vec_normalize import (
        VecNormalize,
    )

    print("Wrapping environment in VecNormalize...")
    env = VecNormalize(
        env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
    )

  print("Initializing PPO model...")

  pi_arch = [int(x) for x in FLAGS.pi_layers]
  vf_arch = [int(x) for x in FLAGS.vf_layers]

  if FLAGS.activation_fn == "ReLU":
    activation_fn = nn.ReLU
  elif FLAGS.activation_fn == "Tanh":
    activation_fn = nn.Tanh
  elif FLAGS.activation_fn == "ELU":
    activation_fn = nn.ELU
  else:
    raise ValueError(f"Unknown activation function: {FLAGS.activation_fn}")

  policy_kwargs = dict(
      log_std_init=FLAGS.log_std_init,
      ortho_init=FLAGS.ortho_init,
      activation_fn=activation_fn,
      net_arch=dict(pi=pi_arch, vf=vf_arch),
  )

  clip_range_vf = FLAGS.clip_range_vf if FLAGS.clip_range_vf >= 0 else None

  model = PPO(
      "MlpPolicy",
      env,
      n_steps=FLAGS.n_steps,
      batch_size=FLAGS.batch_size,
      n_epochs=FLAGS.n_epochs,
      gamma=FLAGS.gamma,
      learning_rate=FLAGS.learning_rate,
      clip_range=FLAGS.clip_range,
      clip_range_vf=clip_range_vf,
      ent_coef=FLAGS.ent_coef,
      vf_coef=FLAGS.vf_coef,
      max_grad_norm=0.5,
      gae_lambda=FLAGS.gae_lambda,
      seed=FLAGS.seed,
      policy_kwargs=policy_kwargs,
      verbose=1,
      tensorboard_log=os.path.join(FLAGS.workdir, "tb_logs"),
  )

  checkpoint_callback = CheckpointCallback(
      save_freq=10000, save_path=FLAGS.workdir, name_prefix="rl_model"
  )

  print(f"Starting training for {FLAGS.total_timesteps} timesteps...")
  model.learn(
      total_timesteps=FLAGS.total_timesteps, callback=checkpoint_callback
  )

  # Save PPO model locally first to avoid ZIP file seek corruption on append-only CNS
  local_model_path = "/tmp/final_model"
  print(f"Saving PPO model locally to {local_model_path}...")
  model.save(local_model_path)

  cns_model_path = os.path.join(FLAGS.workdir, "final_model.zip")
  if FLAGS.workdir.startswith("/cns/"):
    print(
        f"CNS Workdir detected. Uploading local model to {cns_model_path} using shutil..."
    )
    import shutil
    shutil.copyfile(local_model_path + ".zip", cns_model_path)
    print("Upload complete.")
  else:
    if FLAGS.workdir != "/tmp":
      import shutil

      os.makedirs(FLAGS.workdir, exist_ok=True)
      shutil.copyfile(local_model_path + ".zip", cns_model_path)
      print(f"Model moved to local destination: {cns_model_path}")

  if FLAGS.use_vec_normalize:
    local_vn_path = "/tmp/vec_normalize.pkl"
    print(f"Saving VecNormalize state locally to {local_vn_path}...")
    env.save(local_vn_path)

    cns_vn_path = os.path.join(FLAGS.workdir, "vec_normalize.pkl")
    if FLAGS.workdir.startswith("/cns/"):
      print(
          f"CNS Workdir detected. Uploading local VecNormalize to {cns_vn_path} using shutil..."
      )
      import shutil
      shutil.copyfile(local_vn_path, cns_vn_path)
      print("Upload complete.")
    else:
      if FLAGS.workdir != "/tmp":
        import shutil

        os.makedirs(FLAGS.workdir, exist_ok=True)
        shutil.copyfile(local_vn_path, cns_vn_path)
        print(f"VecNormalize moved to local destination: {cns_vn_path}")

  env.close()


if __name__ == "__main__":
  app.run(main)
