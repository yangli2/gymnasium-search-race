r"""Parser script for TensorBoard event logs in gym_search_race using EventAccumulator."""

import sys
from absl import app
from tensorboard.backend.event_processing import event_accumulator


def get_final_metrics(event_file_path):
  try:
    # Size guidance: load only necessary scalar elements to be fast
    size_guidance = {
        event_accumulator.SCALARS: 1000,
    }
    acc = event_accumulator.EventAccumulator(
        event_file_path, size_guidance=size_guidance
    )
    acc.Reload()

    tags = acc.Tags()
    metrics = {"reward": None, "length": None, "step": -1}

    if "scalars" in tags:
      if "rollout/ep_rew_mean" in tags["scalars"]:
        scalars = acc.Scalars("rollout/ep_rew_mean")
        if scalars:
          metrics["reward"] = scalars[-1].value
          metrics["step"] = scalars[-1].step
      if "rollout/ep_len_mean" in tags["scalars"]:
        scalars = acc.Scalars("rollout/ep_len_mean")
        if scalars:
          metrics["length"] = scalars[-1].value
    return metrics
  except Exception as e:
    print(f"Error reading {event_file_path}: {e}", file=sys.stderr)
  return None


def main(argv):
  if len(argv) < 2:
    print("Usage: blaze run :parse_events -- <event_file_path>")
    sys.exit(1)

  path = argv[1]
  metrics = get_final_metrics(path)
  if metrics:
    print(f"Final ep_rew_mean: {metrics['reward']} at step {metrics['step']}")
    print(f"Final ep_len_mean: {metrics['length']}")
  else:
    print("Failed to parse metrics.")


if __name__ == "__main__":
  app.run(main)
