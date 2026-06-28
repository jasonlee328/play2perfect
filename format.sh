#!/bin/bash
# Format the codebase: Ruff (lint-fix imports + format) over the Python packages,
# and xmllint pretty-print over URDFs.
set -e

PKGS="isaacsimenvs evaluation deployment download_checkpoints.py"

for target in $PKGS; do
  echo "Ruff: $target"
  ruff check --extend-select I --fix "$target"
  ruff format "$target"
done

echo "xmllint: assets/**/*.urdf"
find assets -name "*.urdf" -type f -exec xmllint --format "{}" --output "{}" \;

echo "✅ Formatting complete."
