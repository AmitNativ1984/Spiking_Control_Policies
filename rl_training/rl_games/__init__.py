"""rl_games integration: custom network builders, the teacher-student agent, and the runner.

Holds only path constants — importing it must not pull in torch, because `import isaacgym`
has to come first in any entry point (see runner.py). Import the sub-packages explicitly:

    import rl_training.rl_games.networks            # registers the network builders
    from rl_training.rl_games.agents import register_algos
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Where rl_games writes tensorboard logs and checkpoints (train_dir).
RUNS_DIR = os.path.join(REPO_ROOT, "runs")

# Observation statistics and the PopSAN encoder bounds cache derived from them.
OBS_STATS_DIR = os.path.join(RUNS_DIR, "obs_stats")
DEFAULT_BOUNDS_CACHE = os.path.join(OBS_STATS_DIR, "observation_bounds.json")
