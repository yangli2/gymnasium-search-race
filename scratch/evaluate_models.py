import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium_search_race

models_to_eval = [
    {
        "name": "MadPodRacingDiscrete-v2_1",
        "env_id": "gymnasium_search_race/MadPodRacingDiscrete-v2",
        "model_path": "rl-trained-agents/ppo/gymnasium_search_race-MadPodRacingDiscrete-v2_1/best_model.zip",
        "env_kwargs": {"thrust_levels": 2, "pruned_actions": False},
        "map_ids": range(13)
    },
    {
        "name": "MadPodRacingDiscrete-v2_2",
        "env_id": "gymnasium_search_race/MadPodRacingDiscrete-v2",
        "model_path": "rl-trained-agents/ppo/gymnasium_search_race-MadPodRacingDiscrete-v2_2/best_model.zip",
        "env_kwargs": {"thrust_levels": 2, "pruned_actions": False},
        "map_ids": range(13)
    },
    {
        "name": "MadPodRacingBlockerDiscrete-v2_1",
        "env_id": "gymnasium_search_race/MadPodRacingBlockerDiscrete-v2",
        "model_path": "rl-trained-agents/ppo/gymnasium_search_race-MadPodRacingBlockerDiscrete-v2_1/best_model.zip",
        "env_kwargs": {"opponent_path": "rl-trained-agents/ppo/gymnasium_search_race-MadPodRacingDiscrete-v2_1/best_model.zip"},
        "map_ids": range(13)
    },
    {
        "name": "SearchRaceDiscrete-v3_1",
        "env_id": "gymnasium_search_race/SearchRaceDiscrete-v3",
        "model_path": "rl-trained-agents/ppo/gymnasium_search_race-SearchRaceDiscrete-v3_1/best_model.zip",
        "env_kwargs": {},
        "map_ids": [1, 2, 700]
    },
    {
        "name": "SearchRace-v3_1 (Continuous)",
        "env_id": "gymnasium_search_race/SearchRace-v3",
        "model_path": "rl-trained-agents/ppo/gymnasium_search_race-SearchRace-v3_1/best_model.zip",
        "env_kwargs": {},
        "map_ids": [1, 2, 700]
    }
]

print("=== Starting Model Evaluation ===")

for info in models_to_eval:
    print(f"\nEvaluating {info['name']}...")
    try:
        # Load the model
        model = PPO.load(info["model_path"])
        
        lengths = []
        rewards = []
        successes = 0
        
        for map_id in info["map_ids"]:
            kwargs = info["env_kwargs"].copy()
            # For blocker environment, opponent_path might already be specified
            if "opponent_path" not in kwargs:
                kwargs["test_id"] = map_id
            kwargs["sequential_maps"] = False
            
            raw_env = gym.make(info["env_id"], **kwargs)
            env = DummyVecEnv([lambda: raw_env])
            
            obs = env.reset()
            done = False
            ep_len = 0
            ep_rew = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = env.step(action)
                done = dones[0]
                ep_len += 1
                ep_rew += reward[0]
            
            if ep_len < 600:
                successes += 1
                lengths.append(ep_len)
            
            rewards.append(ep_rew)
            env.close()
            
        avg_len = np.mean(lengths) if lengths else 600
        avg_rew = np.mean(rewards)
        print(f"Success Rate: {successes}/{len(info['map_ids'])}")
        if successes > 0:
            print(f"Average completed steps (successes only): {avg_len:.2f}")
        print(f"Average reward: {avg_rew:.2f}")
    except Exception as e:
        print(f"Error evaluating model: {e}")
