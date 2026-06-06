r"""Query status details for a given experiment run."""

from absl import app
from absl import flags

try:
  from experiment_runner.client import runner_api
except ImportError:
  runner_api = None

FLAGS = flags.FLAGS
flags.DEFINE_string("run_id", "", "Experiment Run ID to query")


def main(argv):
  del argv
  if not FLAGS.run_id:
    print("Error: Must specify --run_id.")
    return

  if runner_api is None:
    print("Error: experiment_runner client library is not installed.")
    return

  api = runner_api.RunnerApi()
  exp = api.get_experiment(int(FLAGS.run_id))
  print(f"RUN_STATUS: {exp.status}")
  print(f"RUN_IS_RUNNING: {exp.is_running}")
  print("WORK_UNITS:")
  for wu in exp.get_work_units():
    print(f"  WU {wu.id}: {wu.status.state}")


if __name__ == "__main__":
  app.run(main)
