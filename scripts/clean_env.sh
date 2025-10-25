#!/usr/bin/env bash
set -e

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "❌ Python not found"; exit 1
fi

echo "🧹 Clearing Poetry cache..."
poetry cache clear pypi --all -n || true

echo "🗑 Removing any existing Poetry venv manually..."
rm -rf ~/.cache/pypoetry/virtualenvs/laneatt-* || true

echo "🔧 Recreating env..."
poetry env use "$PY"
poetry install --no-root

echo "✅ Clean reinstall complete."
