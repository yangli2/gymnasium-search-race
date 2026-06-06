r"""Automated exploration and hyperparameter tuning pipeline for gym_search_race."""

import json
import os
import re
import subprocess
import sys
import time
from absl import app
from absl import flags
# No gfile import needed

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "hparams_file",
    "hparams_results.md",
    "Path to the hparams results markdown file",
)
flags.DEFINE_integer("total_timesteps", 10000000, "Timesteps for each run")
flags.DEFINE_integer("iteration", 2, "Current sweep iteration (v2, v3, etc.)")
flags.DEFINE_string(
    "resource_alloc", "", "Resource allocation"
)
flags.DEFINE_bool(
    "monitor",
    True,
    "Whether to monitor the launched experiment to completion.",
)


def parse_hparams_md(file_path):
  """Parses the hparams markdown table into a list of dictionaries."""
  full_path = (
      file_path
      if os.path.isabs(file_path)
      else os.path.join(os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd()), file_path)
  )
  if not os.path.exists(full_path):
    return []

  history = []
  with open(full_path, "r") as f:
    lines = f.readlines()

  # Headers on line 0, separator on line 1
  # Format: | Run | Value | Step | Relative Time | pi_layers | vf_layers | learning_rate | gamma | n_steps | batch_size | n_epochs | clip_range |
  headers = []
  for line in lines:
    if isinstance(line, bytes):
      line = line.decode("utf-8")
    line = line.strip()
    if "|" not in line:
      continue
    parts = [p.strip() for p in line.split("|")]
    # Handle leading/trailing '|'
    if parts[0] == "":
      parts = parts[1:]
    if parts[-1] == "":
      parts = parts[:-1]

    if not parts:
      continue
    if parts[0].startswith("---") or parts[0].startswith("-"):
      continue

    if not headers:
      headers = parts
      continue

    # Parse row
    row = dict(zip(headers, parts))
    try:
      row["Value"] = float(row["Value"])
      row["learning_rate"] = float(row["learning_rate"])
      row["gamma"] = float(row["gamma"])
      row["n_steps"] = int(row["n_steps"])
      row["clip_range"] = float(row["clip_range"])
      history.append(row)
    except (ValueError, KeyError):
      pass

  return history


def get_best_config(history):
  """Gets the historical configuration with the highest reward (Value)."""
  if not history:
    # Return safe fallback baseline
    return {
        "pi_layers": "64,64,64",
        "vf_layers": "256,256,256",
        "learning_rate": 5e-5,
        "gamma": 0.995,
        "n_steps": 4096,
        "clip_range": 0.2,
    }
  # Sort by Value descending
  sorted_hist = sorted(
      history, key=lambda x: x.get("Value", -999.0), reverse=True
  )
  best = sorted_hist[0]
  return {
      "pi_layers": best.get("pi_layers", "64,64,64"),
      "vf_layers": best.get("vf_layers", "256,256,256"),
      "learning_rate": float(best.get("learning_rate", 5e-5)),
      "gamma": float(best.get("gamma", 0.995)),
      "n_steps": int(best.get("n_steps", 4096)),
      "clip_range": float(best.get("clip_range", 0.2)),
  }


def propose_sweep(best, iteration, num_candidates=10):
  """Generates mutated hyperparameter configurations around the best config."""
  import random

  random.seed(time.time())

  sweep = []
  # Always keep the best config as baseline in the new sweep
  sweep.append({"name": f"Iter{iteration}_Baseline", **best})

  # Network options
  net_architectures = [
      ("64,64,64", "128,128,128"),
      ("64,64,64", "256,256,256"),
      ("128,128,128", "256,256,256"),
      ("128,128,128", "512,512,512"),
  ]

  # Mutate to generate remaining candidates
  for i in range(1, num_candidates):
    candidate = best.copy()

    # Mutate learning rate (scale by 0.5 to 2.0)
    lr_factor = random.choice([0.5, 0.75, 1.25, 1.5, 2.0])
    candidate["learning_rate"] = min(
        max(best["learning_rate"] * lr_factor, 1e-5), 5e-4
    )

    # Mutate clip range
    candidate["clip_range"] = random.choice([0.1, 0.2, 0.3, 0.4])

    # Mutate gamma (discount factor)
    candidate["gamma"] = random.choice([0.99, 0.995, 0.998, 0.999])

    # Mutate n_steps
    candidate["n_steps"] = random.choice([2048, 4096, 8192])

    # Randomly mutate network layers
    if random.random() < 0.3:
      pi, vf = random.choice(net_architectures)
      candidate["pi_layers"] = pi
      candidate["vf_layers"] = vf

    name = f"Iter{iteration}_Sweep_{i}"
    sweep.append({"name": name, **candidate})

  return sweep


def write_sweeps_json(sweep, dir_path):
  """Writes the sweep configurations to sweeps.json."""
  sweeps_file = os.path.join(dir_path, "sweeps.json")
  with open(sweeps_file, "w") as f:
    json.dump(sweep, f, indent=2)
  print(f"Successfully wrote sweeps to {sweeps_file}")


def launch_sweep(total_timesteps, resource_alloc, workspace_dir):
  """Launches the sweep training run and extracts the run ID."""
  run_id = str(int(time.time()))
  cmd = [
      sys.executable,
      "launcher.py",
      f"--total_timesteps={total_timesteps}",
  ]
  print(f"Executing launch command: {' '.join(cmd)}")

  # Run command and capture output
  process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
      cwd=workspace_dir,
  )

  for line in process.stdout:
    print(line, end="")

  process.wait()
  if process.returncode != 0:
    raise RuntimeError(f"Launcher failed with exit code {process.returncode}")

  return run_id


def monitor_and_evaluate(run_id, sweep_configs, hparams_file):
  """Monitors the launched experiment and evaluates rewards once complete."""
  print(f"Starting monitoring loop for Run ID {run_id}...")

  completed = False
  runs_dir = f"/tmp/gym_search_race/runs"

  while not completed:
    print(f"Checking file completion under {runs_dir}...")
    all_done = True

    # Check if each sweep candidate has a 'final_model.zip'
    for c in sweep_configs:
      final_model_path = f"{runs_dir}/{c['name']}/final_model.zip"
      if not os.path.exists(final_model_path):
        all_done = False
        print(f"  - {c['name']}: Still running (final_model.zip not found).")
      else:
        print(f"  - {c['name']}: Completed!")

    if all_done:
      completed = True
      print("All sweep candidates have completed training!")
      break

    # Wait 30 minutes before polling again
    print("Waiting 30 minutes before next poll...")
    time.sleep(1800)

  # All done! Now parse the results using our parse_events binary!
  print("Evaluating rewards for all sweep candidates...")
  results = []

  for c in sweep_configs:
    # Find the events log file under tb_logs/PPO_1/
    tb_dir = f"{runs_dir}/{c['name']}/tb_logs/PPO_1"
    try:
      files = os.listdir(tb_dir)
      event_files = [f for f in files if "events.out.tfevents" in f]
      if not event_files:
        print(f"No event files found for {c['name']}")
        results.append({**c, "Value": "CRASH"})
        continue

      event_file_path = os.path.join(tb_dir, event_files[0])

      # Run the parse_events helper to extract the reward
      from parse_events import get_final_metrics
      metrics = get_final_metrics(event_file_path)
      if metrics and metrics["reward"] is not None:
        reward = float(metrics["reward"])
        results.append({**c, "Value": reward})
        print(f"  - {c['name']}: Reward = {reward}")
      else:
        results.append({**c, "Value": "ERROR"})
        print(f"  - {c['name']}: Failed to parse reward output")

    except Exception as e:
      print(f"Error evaluating {c['name']}: {e}")
      results.append({**c, "Value": "CRASH"})

  # Append the new results to hparams_results.md in markdown table format
  print(f"Saving results to {hparams_file}...")
  md_lines = []
  for r in results:
    line = (
        f"| {run_id}/{r['name']} | {r['Value']} | 10092544 | 2.00 |"
        f" {r['pi_layers']} | {r['vf_layers']} | {r['learning_rate']} |"
        f" {r['gamma']} | {r['n_steps']} | - | - | {r['clip_range']} |"
    )
    md_lines.append(line)

  full_hparams_path = os.path.join(
      os.environ.get("BUILD_WORKSPACE_DIRECTORY", ""), hparams_file
  )
  with open(full_hparams_path, "a") as f:
    for l in md_lines:
      f.write(l + "\n")

  print("Auto-tuning iteration successfully completed!")


def main(argv):
  del argv

  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())

  print("=== Reinforcement Learning Auto-Tuner ===")

  # 1. Parse history
  history = parse_hparams_md(FLAGS.hparams_file)
  print(f"Parsed {len(history)} past training evaluations.")

  # 2. Get best config
  best = get_best_config(history)
  print(f"Current best hyperparameter configuration: {best}")

  # 3. Propose next sweep
  sweep = propose_sweep(best, FLAGS.iteration)
  print(
      f"Proposed {len(sweep)} sweep candidate configurations for iteration"
      f" {FLAGS.iteration}:"
  )
  for c in sweep:
    print(
        f"  - {c['name']}: LR={c['learning_rate']}, Clip={c['clip_range']},"
        f" Gamma={c['gamma']}, N_steps={c['n_steps']}"
    )

  # 4. Write sweeps.json
  write_sweeps_json(sweep, workspace_dir)

  # 5. Launch experiment
  try:
    run_id = launch_sweep(
        FLAGS.total_timesteps, FLAGS.resource_alloc, workspace_dir
    )
  except Exception as e:
    print(f"Failed to launch sweep: {e}", file=sys.stderr)
    sys.exit(1)

  # 6. Start monitoring and evaluation
  if FLAGS.monitor:
    monitor_and_evaluate(run_id, sweep, FLAGS.hparams_file)
  else:
    print(
        f"\n>>> Launch completed. Run ID is {run_id}. Skipping synchronous"
        " monitoring. <<<"
    )


if __name__ == "__main__":
  app.run(main)
