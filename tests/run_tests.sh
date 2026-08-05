#!/usr/bin/env bash
# Run the test suite for the refactored tree (config/, task/, rl_training/).
#
# PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 is REQUIRED: the environment ships a broken hypothesis
# pytest plugin whose entry-point scan crashes pytest at startup (TypeError in
# packaging.markers) before any test runs. Disabling plugin autoload sidesteps it; we
# don't rely on any pytest plugins here.
#
# test_task_smoke.py builds a real Isaac Gym sim (16 envs) and needs a GPU; the rest are
# CPU-only. Isaac Gym allows one sim per process, so the sim is a session fixture.
#
# Usage:
#   ./tests/run_tests.sh                      # everything
#   ./tests/run_tests.sh --ignore=tests/test_task_smoke.py   # no GPU
#   ./tests/run_tests.sh -k popsan            # filtered
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v "$@"
