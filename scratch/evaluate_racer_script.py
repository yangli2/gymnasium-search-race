import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

# Add root directory to python path
sys.path.append(os.getcwd())
import racer_script
import gymnasium_search_race

print("=== Evaluating racer_script.py Agent ===")

env_id = "gymnasium_search_race/MadPodRacingDiscrete-v2"
env_kwargs = {"thrust_levels": 3, "pruned_actions": True}
map_ids = range(13)

lengths = []
rewards = []
successes = 0

for map_id in map_ids:
    kwargs = env_kwargs.copy()
    kwargs["test_id"] = map_id
    kwargs["sequential_maps"] = False
    
    raw_env = gym.make(env_id, **kwargs)
    env = DummyVecEnv([lambda: raw_env])
    
    obs = env.reset()
    done = False
    ep_len = 0
    ep_rew = 0
    
    while not done:
        # Note: SB3 VecEnv returns observations wrapped in a batch, i.e. shape (1, 10).
        # We need to extract the first observation (shape (10,)) for the racer script act() function.
        single_obs = obs[0]
        
        # Predict action using racer_script
        action_idx = racer_script.act(single_obs)
        
        # SB3 expects actions as a batch array/list
        obs, reward, dones, infos = env.step([action_idx])
        done = dones[0]
        ep_len += 1
        ep_rew += reward[0]
    
    if ep_len < 600:
        successes += 1
        lengths.append(ep_len)
        print(f"Map {map_id}: Finished in {ep_len} steps, Reward: {ep_rew:.2f}")
    else:
        print(f"Map {map_id}: Timeout (600 steps), Reward: {ep_rew:.2f}")
        
    rewards.append(ep_rew)
    env.close()

avg_len = np.mean(lengths) if lengths else 600
avg_rew = np.mean(rewards)
print("\n=== Summary ===")
print(f"Success Rate: {successes}/13")
if successes > 0:
    print(f"Average completed steps (successes only): {avg_len:.2f}")
print(f"Average reward: {avg_rew:.2f}")
