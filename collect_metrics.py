r"""Collect and parse metrics for completed sweeps inside gym_search_race."""

import json
import os
import re
import subprocess
import sys
from absl import app
from absl import flags
from parse_events import get_final_metrics

FLAGS = flags.FLAGS
flags.DEFINE_string("run_id", "", "Experiment Run ID to evaluate")
flags.DEFINE_string(
    "hparams_file",
    "hparams_results.md",
    "Hparams results file path",
)


def main(argv):
  del argv
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())

  base_dir = f"/tmp/gym_search_race/runs/{FLAGS.run_id}"

  # Dynamically discover all run folders under the base path
  try:
    subdirs = os.listdir(base_dir)
    folders = [
        d.strip("/")
        for d in subdirs
        if os.path.isdir(os.path.join(base_dir, d))
    ]
  except Exception as e:
    print(f"Error listing directories in {base_dir}: {e}", file=sys.stderr)
    sys.exit(1)

  if not folders:
    print(
        f"Error: No sweep subdirectories found under {base_dir}",
        file=sys.stderr,
    )
    sys.exit(1)

  results = []
  print(f"=== Extracting Rewards for Run ID {FLAGS.run_id} ===")

  for name in folders:
    path_pattern = f"{base_dir}/{name}/tb_logs/PPO_1/events.out.tfevents.*"
    try:
      import glob
      files = glob.glob(path_pattern)
      if not files:
        print(f"  - {name}: NO EVENT LOGS FOUND")
        results.append({"name": name, "reward": "CRASH"})
        continue

      event_file = files[0]

      # Call direct Python API helper
      metrics = get_final_metrics(event_file)
      if metrics and metrics["reward"] is not None:
        results.append({
            "name": name,
            "reward": metrics["reward"],
            "length": metrics["length"],
        })
        print(
            f"  - {name}: {metrics['reward']} (len: {metrics['length']}) at"
            f" step {metrics['step']}"
        )
      else:
        results.append({
            "name": name,
            "reward": "PARSE_ERROR",
            "length": "PARSE_ERROR",
        })
        print(f"  - {name}: PARSE ERROR")

    except Exception as e:
      print(f"  - {name}: ERROR: {e}")
      results.append({"name": name, "reward": "ERROR", "length": "ERROR"})

  # Sort by reward (highest first)
  def sort_key(x):
    r = x["reward"]
    if isinstance(r, float):
      return (0, -r)
    return (1, r)

  sorted_results = sorted(results, key=sort_key)

  print("\n\n================================================================")
  print(f"### SWEEP RESULTS SUMMARY (Run ID {FLAGS.run_id})")
  print("================================================================")

  print(
      "| Configuration Sweep Name | Value (Final ep_rew_mean) | Length (Final"
      " ep_len_mean) |"
  )
  print("| :--- | :---: | :---: |")
  for r in sorted_results:
    val_str = (
        f"**{r['reward']:.3f}**"
        if isinstance(r["reward"], float)
        else f"*{r['reward']}*"
    )
    len_str = (
        f"{r['length']:.1f}"
        if isinstance(r["length"], float)
        else f"*{r['length']}*"
    )
    print(f"| {r['name']} | {val_str} | {len_str} |")
  print("----------------------------------------------------------------\n")

  # Load sweeps.json to map run names to their hyperparameter coordinates
  sweeps_path = os.path.join(
      workspace_dir, "sweeps.json"
  )
  sweeps_dict = {}
  if os.path.exists(sweeps_path):
    print(f"Loading sweeps from {sweeps_path} to extract parameters...")
    try:
      with open(sweeps_path, "r") as f:
        sweeps_list = json.load(f)
        sweeps_dict = {hp["name"]: hp for hp in sweeps_list}
    except Exception as e:
      print(f"Warning: Failed to load/parse sweeps.json: {e}")

  # Append to hparams_results.md
  hparams_path = os.path.join(workspace_dir, FLAGS.hparams_file)
  print(f"Updating {hparams_path}...")
  md_lines = []
  for r in sorted_results:
    hp = sweeps_dict.get(r["name"], {})
    pi = hp.get("pi_layers", "-")
    vf = hp.get("vf_layers", "-")
    lr = hp.get("learning_rate", "-")
    gamma = hp.get("gamma", "-")
    n_steps = hp.get("n_steps", 4096)
    batch = hp.get("batch_size", "-")
    epochs = hp.get("n_epochs", "-")
    clip = hp.get("clip_range", "-")

    line = (
        f"| {FLAGS.run_id}/{r['name']} | {r['reward']} | 10092544 | 2.00 | {pi} |"
        f" {vf} | {lr} | {gamma} | {n_steps} | {batch} | {epochs} | {clip} |"
    )
    md_lines.append(line)

  with open(hparams_path, "a") as f:
    for l in md_lines:
      f.write(l + "\n")
  print("Update completed successfully.")


if __name__ == "__main__":
  # We need import os inside main to have it resolved
  import os

  app.run(main)
