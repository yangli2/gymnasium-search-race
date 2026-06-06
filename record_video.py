r"""Record a video of a trained gym_search_race agent."""

import json
import os
import sys
from absl import app
from absl import flags
import gymnasium as gym

# Register environments in gym
import gymnasium_search_race  # pylint: disable=unused-import
from stable_baselines3 import PPO

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", "", "Path to trained PPO model .zip")
flags.DEFINE_string(
    "env_id",
    "gymnasium_search_race/MadPodRacingDiscrete-v2",
    "Environment ID",
)
flags.DEFINE_string(
    "video_folder",
    "/tmp/gym_search_race_videos",
    "Folder to save videos",
)
flags.DEFINE_string("env_kwargs_json", "{}", "JSON string of env kwargs")
flags.DEFINE_string("opponent_path", "", "Path to opponent PPO model")


def record_video(model_path, env_id, video_folder, env_kwargs, opponent_path):
  # Ensure env kwargs has sequential_maps=True by default for realistic test
  env_kwargs = env_kwargs or {}
  if "sequential_maps" not in env_kwargs:
    env_kwargs["sequential_maps"] = True

  print(f"Creating environment {env_id} with kwargs: {env_kwargs}")

  opponent_path_arg = opponent_path if opponent_path else None

  # We need to import pygame before making env if pygame is used for rendering
  import pygame

  env = gym.make(
      env_id,
      render_mode="rgb_array",
      opponent_path=opponent_path_arg,
      **env_kwargs,
  )

  local_model_path = model_path
  local_vn_path = None

  vn_local_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
  if os.path.exists(vn_local_path):
    local_vn_path = vn_local_path

  from stable_baselines3.common.vec_env import DummyVecEnv

  env = DummyVecEnv([lambda: env])

  if local_vn_path is not None:
    print(f"Loading VecNormalize statistics from {local_vn_path}...")
    from stable_baselines3.common.vec_env import VecNormalize

    env = VecNormalize.load(local_vn_path, env)
    env.training = False
    env.norm_reward = False

  print(f"Loading model from {local_model_path}...")
  model = PPO.load(path=local_model_path, env=env)

  observation = env.reset()
  done = False
  step_count = 0
  frames = []

  print("Running episode...")
  from PIL import Image

  while not done:
    # Capture frame
    frame = env.render()
    if frame is not None:
      frames.append(Image.fromarray(frame))

    action, _ = model.predict(observation=observation, deterministic=True)
    observation, reward, dones, infos = env.step(action)
    done = dones[0]
    step_count += 1

  # Capture final frame
  frame = env.render()
  if frame is not None:
    frames.append(Image.fromarray(frame))

  print(f"Episode finished in {step_count} steps.")

  if frames:
    final_gif_path = os.path.join(video_folder, "trajectory.gif")
    os.makedirs(video_folder, exist_ok=True)
    print(f"Saving {len(frames)} frames as GIF to {final_gif_path}...")
    frames[0].save(
        final_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,  # 50ms per frame = 20 FPS
        loop=0,
    )
    print(f"GIF saved to {final_gif_path}.")
  else:
    print("Error: No frames captured!")

  env.close()


def main(argv):
  del argv
  if not FLAGS.model_path:
    print("Error: --model_path is required.", file=sys.stderr)
    sys.exit(1)

  env_kwargs = json.loads(FLAGS.env_kwargs_json)
  record_video(
      model_path=FLAGS.model_path,
      env_id=FLAGS.env_id,
      video_folder=FLAGS.video_folder,
      env_kwargs=env_kwargs,
      opponent_path=FLAGS.opponent_path,
  )


if __name__ == "__main__":
  app.run(main)
