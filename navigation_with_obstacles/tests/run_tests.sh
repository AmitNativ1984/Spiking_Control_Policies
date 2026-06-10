#!/usr/bin/env bash
# Run the navigation_with_obstacles test suite.
#
# PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 is REQUIRED: the environment ships a broken
# hypothesis pytest plugin whose entry-point scan crashes pytest at startup
# (TypeError in packaging.markers) before any test runs. Disabling plugin
# autoload sidesteps it; we don't rely on any pytest plugins here.
#
# The suite builds a full Isaac Gym sim once (shared session fixture in
# conftest.py) and needs a GPU.
#
# Usage:
#   ./navigation_with_obstacles/tests/run_tests.sh            # all tests
#   ./navigation_with_obstacles/tests/run_tests.sh -k stale   # filtered
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest navigation_with_obstacles/tests/ -v "$@"
