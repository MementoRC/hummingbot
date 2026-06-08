#!/usr/bin/env bash
# Fast diff-driven test run vs origin/ci-base.
# Selects tests touched by the branch's diff, then runs pixi pytest.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SELECTED="$(mktemp)"
trap 'rm -f "$SELECTED"' EXIT

python3 .github/test-selection/select_tests.py \
  --mode branch \
  --base-ref origin/ci-base \
  --head-ref HEAD \
  --config .github/test-selection/test-selection-map.yaml \
  --repo . \
  --no-history > "$SELECTED"

if [ ! -s "$SELECTED" ]; then
  echo "No tests touched by diff — quick check passes."
  exit 0
fi

echo "Selected $(wc -l < "$SELECTED") test files:"
cat "$SELECTED"
echo "---"

exec pixi run -e ci pytest $(cat "$SELECTED") -v --tb=short
