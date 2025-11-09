#!/usr/bin/env bash
set -euo pipefail

# Notes:
# - This script is robust against a currently-activated (and possibly broken) Poetry venv.
# - It will remove any existing envs for this project, then recreate and install deps.

# Resolve a Python interpreter to base the new venv on
PY="$(command -v python3 || command -v python || true)"
if [ -z "${PY:-}" ]; then
  echo "❌ Python not found"; exit 1
fi

# Ensure we don't run Poetry inside a broken/removed venv
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "🚪 Detected active virtualenv: $VIRTUAL_ENV"
  echo "   Temporarily unsetting it for this script to avoid path issues..."
  # Try to remove the venv's bin dir from PATH and unset VIRTUAL_ENV for this subshell
  PATH="${PATH//:$VIRTUAL_ENV\/bin/}"
  PATH="${PATH//$VIRTUAL_ENV\/bin:/}"
  PATH="${PATH//$VIRTUAL_ENV\/bin/}"
  export PATH
  unset VIRTUAL_ENV
  hash -r || true
fi

echo "🧹 Clearing Poetry cache..."
poetry cache clear pypi --all -n || true

echo "🗑 Removing any existing Poetry venv(s) for this project..."
# First, ask Poetry to remove them cleanly (non-interactive)
poetry env remove --all -n >/dev/null 2>&1 || true

# As a fallback, remove any matching venv directories for this project name
# Try to infer project name from Poetry; if not available, fallback to common prefixes
PROJECT_NAME="$(poetry version --no-ansi 2>/dev/null | awk '{print $1}' || true)"
if [ -n "${PROJECT_NAME:-}" ]; then
  rm -rf "$HOME/.cache/pypoetry/virtualenvs/${PROJECT_NAME}-"* || true
else
  rm -rf "$HOME/.cache/pypoetry/virtualenvs/laneatt-"* || true
  rm -rf "$HOME/.cache/pypoetry/virtualenvs/laneatt-classproj-"* || true
fi

echo "🔧 Recreating env with: $PY"
poetry env use "$PY"
poetry install --no-root

echo "✅ Clean reinstall complete."
