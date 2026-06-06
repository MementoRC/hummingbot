# Hummingbot Modularization Guide

> Static reference for the bleeding-edge fork's sub-package architecture.
> Read this first if you are a returning solo developer or a new contributor.

---

## TL;DR

This fork of [hummingbot/hummingbot](https://github.com/hummingbot/hummingbot) is being progressively carved
into independent `hb-*` sub-packages hosted under `MementoRC/hb-<name>` and mounted as git submodules at
`sub-packages/<name>/`. Each sub-package owns a well-defined slice of the monolith (data types, connectors,
loggers, candles feed, etc.), ships its own `pyproject.toml`, test suite, and pixi environment, and can be
developed and released independently of the parent repo.

Sub-packages exist to isolate regressions (a change to data-type-primitives cannot silently break the candles
feed without a visible API boundary), enable thorough isolated tests, allow strict dependency review at package
granularity, and give a solo developer bounded flexibility to refactor one layer at a time. The cron-managed
rebuild pipeline (`custom_git_setup/scripts/hummingbot-cron-wrapper.sh`) merges tracked `_for_bleed/*` branches
into `bleeding-edge` nightly, so individual feature branches stay small and reviewable. All file-modifying work
on feature branches happens in git worktrees — never by checking out into the cron-managed
`hummingbot/` directory.

---

## The 14 Sub-packages

| Name | Location | Supersedes (hummingbot module) | Hard dep on another hb-* | Purpose |
|---|---|---|---|---|
| `async-utils` | `sub-packages/async-utils/` | (not declared) | none | Shared async helpers and task utilities |
| `candles-feed` | `sub-packages/candles-feed/` | `hummingbot.data_feed.candles_feed` | none | OHLCV candle data ingestion across exchanges |
| `connector-utils` | `sub-packages/connector-utils/` | (not declared) | none | Low-level connector plumbing (auth, REST, WS) |
| `data-type-primitives` | `sub-packages/data-type-primitives/` | `data_type_primitives` (core data types) | none | Primitive exchange data types (orders, trades, fees, funding) |
| `event-bus` | `sub-packages/event-bus/` | (not declared) | none | Typed pub/sub event bus (extracted from pubsub.pyx) |
| `liquidations-feed` | `sub-packages/liquidations-feed/` | (not declared) | none | Real-time liquidation event feed |
| `logger` | `sub-packages/logger/` | `hummingbot.logger` (most modules; log_server_client excluded) | none | HummingbotLogger, CLI handler, application warnings |
| `market-connector` | `sub-packages/market-connector/` | `hummingbot.connector` | none | Exchange gateway framework and connector base classes |
| `market-data` | `sub-packages/market-data/` | (not declared) | none | Orderbook and trade tape aggregation |
| `market-simulator` | `sub-packages/market-simulator/` | (not declared) | none | Paper-trading and backtesting simulation layer |
| `rate-oracle` | `sub-packages/rate-oracle/` | (not declared) | none | Exchange rate resolution for portfolio valuation |
| `remote-iface` | `sub-packages/remote-iface/` | (not declared) | none | MQTT / commlib-py remote control interface |
| `strategy-framework` | `sub-packages/strategy-framework/` | `hummingbot.strategy_v2` | none | Strategy v2 controller/executor framework |
| `web-assistant` | `sub-packages/web-assistant/` | (not declared) | none | HTTP + WebSocket client wrappers for connector use |

**Reading the "supersedes" column**: the `[tool.hummingbot.supersedes]` table in each sub-package's
`pyproject.toml` is the machine-readable source of truth. When a field reads "(not declared)", the sub-package
has not yet published its migration metadata — it is either very early-stage or the metadata authoring is
pending.

---

## Architecture: hb_compat Layer

Each sub-package that replaces a `hummingbot.*` module may ship an `hb_compat/` subdirectory (or an
`hb_compat.py` inline shim). This is the **only** location inside a sub-package that is permitted to import
from `hummingbot.*`. The rest of the sub-package must import only from the sub-package's own namespace and
from other `hb-*` packages — never from the monolith directly.

Two patterns currently coexist:

**Pattern A — dedicated subdirectory** (`candles_feed/hb_compat/`):
```
sub-packages/candles-feed/
    candles_feed/
        hb_compat/          # imports hummingbot.data_feed.candles_feed.*
            __init__.py
            adapters.py
```

**Pattern B — inline try/except in the consuming module**:
```python
try:
    from hummingbot.strategy_v2 import SomeClass
except ImportError:
    from strategy_framework import SomeClass  # standalone fallback
```

Pattern A is the target. Pattern B is a known inconsistency left over from early extraction work. Standardizing
to Pattern A across all sub-packages is tracked but not yet scheduled. When reading sub-package source, treat
any direct `from hummingbot.*` import outside of `hb_compat/` as a tech-debt marker.

The `[tool.hummingbot.supersedes]` `import_path` field records which sub-package module provides the compat
shim (e.g., `"candles_feed.hb_compat"` for candles-feed, `"strategy_framework"` for strategy-framework).

---

## Branch Model

The fork uses a four-tier linear stack:

```
upstream/development  (read-only; hummingbot/hummingbot main line)
       |
       v  sync (periodic)
development           (fork of upstream; receives upstream syncs, no direct feature work)
       |
       v  merge + ruff format + conflict gate
ci-base               (permanent formatted base layer; also receives _for_ci/* merges)
       |
       v  merge tracked _for_bleed/* branches (ordered by tier)
bleeding-edge         (integration layer; always rebuildable from ci-base + config)
```

**Naming conventions**

| Prefix | Purpose | Example |
|---|---|---|
| `_for_bleed/<name>` | Feature or fix destined for bleeding-edge | `_for_bleed/executor-mixins` |
| `_for_bleed_manual/<name>` | Manual-only patch; not auto-merged by cron | `_for_bleed_manual/kraken-tweaks` |
| `_for_ci/<name>` | Infrastructure, CI, tooling, docs for ci-base | `_for_ci/modularization-docs` |

**Authoritative source**: `custom_git_setup/configs/branch-tracking.yaml` lists every tracked branch, its
merge tier, and any parent-branch dependencies. Do not maintain a separate list of branches elsewhere — the
config file is the single source of truth.

---

## Cron Rebuild Pipeline

Two scripts drive the automated rebuild:

`custom_git_setup/scripts/hummingbot-branch-tracking.sh --rebuild`

This script:
1. Fetches `development` from the upstream fork.
2. Rebuilds `ci-base` by merging `development`, running `ruff format`, and applying a conflict gate:
   - Logical conflicts (changes to `pyproject.toml`, `conftest.py`, `.github/`, `.pre-commit-config.yaml`) abort
     the rebuild and require manual resolution.
   - Format-only conflicts are auto-resolved with `--theirs` followed by `ruff format`.
3. Merges each `_for_ci/*` branch into ci-base in declaration order.
4. Rebuilds `bleeding-edge` from ci-base, then merges each tracked `_for_bleed/*` branch in tier order
   (infrastructure tier first, feature tier second). Parent-chained branches are merged after their declared
   parent.
5. Pushes `ci-base` and `bleeding-edge` to origin.

`custom_git_setup/scripts/hummingbot-cron-wrapper.sh`

Wraps the rebuild script with:
- Lock file to prevent concurrent runs.
- Log rotation to `custom_git_setup/logs/`.
- Summary write to `custom_git_setup/logs/latest_summary.txt`.
- Optional desktop notification (`--notify` flag; opt-in).

**Operator signal**: `latest_summary.txt` is the primary indicator of rebuild health. Check it after a cron
cycle to confirm all branches merged cleanly and both `ci-base` and `bleeding-edge` were pushed.

---

## Worktree Workflow (MANDATORY)

The `hummingbot/` directory is cron-managed. Its `HEAD` must always point to either `ci-base` or
`bleeding-edge`. Checking out any `_for_bleed/*` or `_for_ci/*` branch directly into `hummingbot/` will
break the next cron rebuild.

**Always use a linked worktree for feature branch work:**

```bash
# Create a worktree for a feature branch
git worktree add ../hummingbot-_for_bleed-<name> _for_bleed/<name>

# Or for a ci branch
git worktree add ../hummingbot-_for_ci-<name> _for_ci/<name>

# Work, edit, format, lint, commit — all inside the worktree directory
cd ../hummingbot-_for_bleed-<name>
pixi run format
pixi run lint
git add -p
git commit -S -m "feat: ..."

# When done, remove the worktree
git worktree remove ../hummingbot-_for_bleed-<name>
```

The branch lives in the shared `.git/` object store — the cron rebuild finds it by name via
`branch-tracking.yaml` regardless of whether a worktree is checked out.

**Convention**: worktree sibling paths use the pattern `../hummingbot-<branch-name-with-slashes-as-dashes>`,
e.g., `../hummingbot-_for_bleed-executor-mixins`.

---

## Validation Workflow for a Solo Developer

Run these pixi tasks before pushing any branch:

```bash
# Format and lint the main repo
pixi run format      # ruff format + isort
pixi run lint        # ruff check (F, E9, S, B rules)

# Full check (format + lint combined gate)
pixi run check

# Sub-package compatibility check (verifies hb_compat shims load cleanly)
pixi run compat-check

# Run all sub-package test suites (manual; slow)
pixi run test-all-subpackages

# Emit the dependency DAG (dot format)
pixi run dep-graph

# Run tach boundary lint (import boundary enforcement)
pixi run lint-boundaries
```

**For a cross-cutting change** (e.g., renaming a type that appears in the main repo AND sub-packages):

1. Make the change on the relevant `_for_bleed/*` branch in its worktree.
2. `pixi run format && pixi run lint` in the worktree.
3. `pixi run compat-check` — confirms hb_compat shims still load.
4. If the change touches a sub-package directly, `cd sub-packages/<name>/ && pixi run test` in that
   sub-package's own environment.
5. `pixi run lint-boundaries` — confirms no new illegal cross-layer imports.
6. Commit (GPG-signed), push the `_for_bleed/*` branch; wait for cron rebuild to integrate.

**Note**: `pixi run compat-check` and `pixi run test-all-subpackages` are defined in the root `pyproject.toml`
tasks block. If a task is missing, add it there — do not call `pytest` or `ruff` directly.

---

## Dual-Import-Path Migration Status

`hb-data-type-primitives` extraction is complete: all six core modules (common, cancellation_result,
funding_info, trade_fee, in_flight_order, limit_order) have been extracted and the sub-package is published.
However, approximately 600 import sites across the main repo and the other 13 sub-packages still reference the
old paths:

```python
# Old path (still works via monolith; not yet removed)
from hummingbot.core.data_type.in_flight_order import InFlightOrder

# New path (canonical; use this in new code)
from data_type_primitives.in_flight_order import InFlightOrder
```

The bulk rewrite (using `libcst` codemod) is pending. Until it lands, **both import paths coexist and are
valid**. When grepping for "where is InFlightOrder used", you must check both paths — a search for only
`from hummingbot.core.data_type` will miss any files already migrated, and a search for only
`from data_type_primitives` will miss the unrewritten majority.

The `[tool.hummingbot.supersedes]` table in each sub-package's `pyproject.toml` is the machine-readable
migration ledger. When `modules` (or `module`) is declared there, the extraction is done but the consumer
rewrite may not be.

A `LimitOrderStatus` identity shim was added in 2026-05-27 to maintain backward compatibility during the
transition. Remove it after the bulk rewrite completes.

---

## Where to Put New Work

Use this decision tree:

```
New work type?
|
+-- Net-new feature for bleeding-edge
|       -> Branch: _for_bleed/<feature-name>
|          Worktree: ../hummingbot-_for_bleed-<feature-name>
|          Add to: custom_git_setup/configs/branch-tracking.yaml (feature tier)
|
+-- Infrastructure / CI / tooling / docs
|       -> Branch: _for_ci/<infra-name>
|          Worktree: ../hummingbot-_for_ci-<infra-name>
|          Merges into: ci-base (not bleeding-edge directly)
|
+-- New sub-package extraction
|       -> Follow the cascade pattern:
|          1. Create MementoRC/hb-<name> repo + pyproject.toml
|          2. Copy source; add [tool.hummingbot.supersedes]
|          3. Atomic commit: add submodule + hb_compat shim in one _for_bleed/* branch
|          4. Follow-up branch: update consumers in main repo
|          5. Add sub-package to sub-packages/ list in this document
|
+-- Manual-only patch (not safe to auto-merge)
        -> Branch: _for_bleed_manual/<patch-name>
           Do NOT add to branch-tracking.yaml auto-merge list
           Merge manually before each bleeding-edge push
```

**Sub-package extraction ordering**: extract leaves first (packages with no hb-* dependencies), then
progressively extract their consumers. The current dependency graph has no cycles; maintain that invariant.

---

## Known Gaps and Open Work

**tach boundary rules**: `pyproject.toml` has tach configured but the per-module boundary rules are not yet
fully populated. `pixi run lint-boundaries` may pass vacuously until rules are declared. Populating rules is
a prerequisite for enforcing the hb_compat layering constraint automatically.

**correlation_id threading**: Executor actions and connector calls do not yet share a correlation ID for
end-to-end request tracing. This is a design gap across the `_for_bleed/executor-mixins` and
`market-connector` layers.

**Sub-package CI rollup**: each sub-package has its own GitHub Actions workflow, but there is no consolidated
rollup job that gates the parent-repo `bleeding-edge` push on all sub-package CI results. This is tracked but
not yet designed.

**hb_compat parity tests**: shims in `hb_compat/` should have tests verifying that the shim surface matches
the monolith surface. These parity tests do not exist yet for most sub-packages.

**Dual-import-path bulk rewrite**: approximately 600 sites still import from `hummingbot.core.data_type.*`
instead of `data_type_primitives.*`. The libcst codemod pass is pending (see section above).

**executor-mixin migration**: `_for_bleed/executor-mixins` (7 mixins, 67 tests) is implemented in
`hummingbot/strategy_v2/executors/mixins`. Port B moves this into `sub-packages/strategy-framework` with full
test coverage, then retires the `_for_bleed/executor-mixins` branch.

**Full audit reference**: `hummingbot_ai_docs/modularization-review-2026-06-01/ROUND3.md` contains the
complete audit findings that informed this document.

---

## Reference Paths

| Purpose | Path |
|---|---|
| Main repository root | `/home/memento/PycharmProjects/Hummingbot/hummingbot/` |
| Sub-packages (git submodules) | `sub-packages/<name>/` relative to repo root |
| Branch tracking config | `custom_git_setup/configs/branch-tracking.yaml` |
| Cron rebuild script | `custom_git_setup/scripts/hummingbot-branch-tracking.sh` |
| Cron wrapper script | `custom_git_setup/scripts/hummingbot-cron-wrapper.sh` |
| Cron logs directory | `custom_git_setup/logs/` |
| Rebuild health summary | `custom_git_setup/logs/latest_summary.txt` |
| AI session sidecar docs | `hummingbot_ai_docs/` (not tracked in bleeding-edge; local only) |
| Sub-package upstream repos | `https://github.com/MementoRC/hb-<name>` |
| `.gitmodules` (submodule pins) | `.gitmodules` in repo root (owned by ci-base) |
| This document | `MODULARIZATION.md` in repo root |

---

*Last updated: 2026-06-01. Maintained on `_for_ci/modularization-docs`; merges into `ci-base` and
subsequently into `bleeding-edge` via the cron rebuild.*
