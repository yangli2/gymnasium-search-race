# Notes and Learnings

This file contains general knowledge, caveats, and learned lessons for working with the `gymnasium_search_race` codebase.

## 1. Running Sweeps and Experiments

### Local Execution

To run hyperparameter tuning sweeps, you can use the local parallel launcher:
```bash
python launcher.py --total_timesteps=100
```
This script will read sweep definitions from `sweeps.json` (or fall back to defaults) and execute parallel training processes locally. Logs and models are saved under `/tmp/gym_search_race/runs`.

## 2. Codebase Conventions and Gotchas

### PyTorch Imports

Be careful with PyTorch imports. Always use standard PyTorch import paths, for example:
```python
from torch import nn
```

### Virtual Environment

Always run commands and scripts within the local virtual environment where the dependencies (like Gymnasium, Pygame, Stable-Baselines3, PyTorch) are installed.
To set up and activate the virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[testing]
```
