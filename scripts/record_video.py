import argparse
import json

import gymnasium as gym
from stable_baselines3 import PPO


def record_video(
    model_path: str,
    env_id: str,
    video_folder: str = "videos",
    opponent_path: str | None = None,
    env_kwargs: dict | None = None,
) -> None:
  print(f'env_kwargs: {env_kwargs}')
  env = gym.make(
      env_id,
      opponent_path=opponent_path,
      render_mode="rgb_array",
      **env_kwargs,
  )

  env = gym.wrappers.RecordVideo(
      env,
      video_folder=video_folder,
      episode_trigger=lambda _: True,
      disable_logger=True,
  )

  model = PPO.load(path=model_path, env=env)

  observation, _info = env.reset()
  terminated = truncated = False

  while not terminated and not truncated:
    action, _ = model.predict(observation=observation, deterministic=True)
    observation, _reward, terminated, truncated, _info = env.step(action)
    print(observation)

  env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Record a video of a trained agent",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--path",
      required=True,
      help="path to model file",
  )
  parser.add_argument(
      "--opponent-path",
      help="path to opponent model file",
  )
  parser.add_argument(
      "--env",
      default=(
          "gymnasium_search_race:gymnasium_search_race/MadPodRacingDiscrete-v2"
      ),
      help="environment id",
  )
  parser.add_argument(
      "--env-kwargs",
      default="",
      type=json.loads,
      help=(
          "A json string encoding a kwargs dictionary to pass into the"
          " environment creation."
      ),
  )
  parser.add_argument(
      "--video-folder",
      default="videos",
      help="path to videos folder",
  )
  args = parser.parse_args()

  record_video(
      model_path=args.path,
      env_id=args.env,
      video_folder=args.video_folder,
      opponent_path=args.opponent_path,
      env_kwargs=args.env_kwargs,
  )