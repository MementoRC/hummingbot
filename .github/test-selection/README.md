# .github/test-selection

## Purpose

Diff-driven test selection for the hummingbot repo. Given a git diff (by ref
pair or diff file), `select_tests.py` resolves which test files should run
using the rules in `test-selection-map.yaml`, narrowing the test surface
without skipping mandatory baseline coverage.

Exit codes: 0 = selection ready, 1 = error, 2 = escape to full suite.

## Used by

- `quick-check.yml` — fast PR/push CI check on `_for_bleed/**` and `_for_ci/**` branches
- `pixi run test-changed` — local iteration: select + run tests for the current branch diff
- `custom_git_setup` cron pipeline — upstream gate verification (Phase 5B will update paths)

## CLI summary

```
python3 .github/test-selection/select_tests.py \
  --mode branch \
  --base-ref origin/ci-base \
  --head-ref HEAD \
  --config .github/test-selection/test-selection-map.yaml \
  --repo .
```

`--mode upstream` diffs against the last synced upstream SHA (stored in state file).
`--shadow` emits selection + marker line but never exits 2 (for dry-run observation).
`--no-history` skips commit dedup and state-file writes (used in CI).

## Test command

```
cd .github/test-selection && python3 -m pytest tests/ -v
```

## Plan doc reference

`hummingbot_ai_docs/2026-06-06-diff-driven-test-selection.md`
