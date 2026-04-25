# Migration Plan: Merge hb-coinbase-connector into hb-market-connector

**Date**: 2026-04-25
**Author**: orchestrated via Claude Code
**Status**: APPROVED — proceeding to Phase A
**Supersedes**: PR #1 on MementoRC/hb-coinbase-connector
**Related**: 2026-04-24-coinbase-connector-implementation.md

---

## 1. Motivation

`hb-coinbase-connector` inherits rich behavior from `hb-market-connector` (`RestConnectorBase`, `WsConnectorBase`, primitives, exceptions, testing infrastructure). The hb_compat inversion-of-control pattern doesn't apply because:

- Dependency direction is wrong (market-connector defines the contract, coinbase satisfies it)
- Parents are rich behavior (TokenBucket, httpx, retry loops), not thin protocols

The CI failures we just fought (path deps, git URL workaround, hatchling `allow-direct-references`, security.yml redundancy) are symptoms of an incorrect package boundary. coinbase-connector is not a sibling — it is a **plugin/specialization** of market-connector.

Reference: ccxt, freqtrade, and similar multi-exchange libraries all ship exchanges as sub-packages of a single distribution, gated by optional-dependencies.

## 2. Goal

Move `coinbase_connector` Python package into `market_connector/exchanges/coinbase/`, ship as `pip install hb-market-connector[coinbase]`, archive `hb-coinbase-connector` repo, retire the parent submodule.

**Out of scope** (deferred to future plans):
- Adding additional exchanges (binance/kraken/okx) — establish the pattern first
- Refactoring transport.py to inversion-of-control composition — orthogonal
- Publishing hb-market-connector to PyPI/conda-forge — orthogonal

## 3. Target Structure

hb-market-connector/ (MementoRC/hb-market-connector)
  market_connector/
    transport/                            (existing — rest_base, ws_base, endpoint, token_bucket)
    primitives.py                         (existing)
    exceptions.py                         (existing)
    testing/                              (existing — mock_transport, contract)
    exchanges/                            (NEW)
      __init__.py
      coinbase/                           (moved from coinbase_connector/)
        __init__.py
        __about__.py                      (independent version: market-connector v0.X.Y, coinbase exchange v0.M.N)
        auth.py
        coinbase_gateway.py
        config.py
        converters.py
        endpoints.py
        transport.py
        mixins/                           (entire dir)
        schemas/                          (entire dir)
        tools/                            (entire dir — fixture_recorder)
  tests/
    unit/                                 (existing market_connector tests)
    exchanges/                            (NEW — house exchange-specific tests)
      coinbase/
        unit/                             (test_transport, test_auth, test_*_mixin, etc.)
        conftest.py
        test_contract.py
    conftest.py                           (existing)
  pyproject.toml
    [project.optional-dependencies]
    coinbase = ["pyjwt>=2.8", "cryptography>=41"]
    dev      = [...]                      (existing — extended to test all exchanges)

Import migration:
- `from coinbase_connector.X` → `from market_connector.exchanges.coinbase.X`
- `from market_connector.X` → unchanged (now intra-package)

## 4. Phases

### Phase A — Pre-migration audit (≈30 min)

A.1 Verify no external consumers:
- Confirm `coinbase_connector` is imported nowhere outside its own sub-package (already verified — only the bleed branch's docs and rate-oracle file reference the *name*, not imports).
- Confirm no PRs/branches in flight on hb-coinbase-connector beyond `feat/phase-10-integration` (PR #1).

A.2 Snapshot current state:
- Tag `feat/phase-10-integration` HEAD on hb-coinbase-connector as `archive/pre-merge-into-market-connector` for rollback.
- Save list of all commits on `feat/phase-10-integration` for credit preservation.

A.3 History-preservation strategy (LOCKED DECISION)

**SELECTED: A1 (git filter-repo)**

Tool: Install via `pixi global install git-filter-repo` or `pip install git-filter-repo`

Process: Each commit on hb-coinbase-connector is rewritten so all paths are prefixed with `market_connector/exchanges/coinbase/` (and tests prefixed similarly). Then merged via `git merge --allow-unrelated-histories` into hb-market-connector.

Rationale: Clean history, full author/date preservation. Modest tooling cost. Most accurate for archeology.

Alternative A2 (git subtree merge): Adds hb-coinbase-connector as a subtree. Trade-off: verbose merge commits, less ergonomic.

Alternative A3 (squash + handcraft): Single squash + follow-up commits per area. Trade-off: loses individual attribution, worst for history.

### Phase B — Code migration on hb-market-connector (≈2 hours)

Branch: `feat/exchanges-coinbase` off `development`.

B.1 Apply path rewrite (A1 strategy):
```
# In a temp clone of hb-coinbase-connector at feat/phase-10-integration:
git filter-repo \
  --path-rename coinbase_connector/:market_connector/exchanges/coinbase/ \
  --path-rename tests/:tests/exchanges/coinbase/

# Merge into hb-market-connector:
cd ../hb-market-connector
git remote add coinbase-archive ../hb-coinbase-connector
git fetch coinbase-archive
git checkout -b feat/exchanges-coinbase development
git merge --allow-unrelated-histories coinbase-archive/feat/phase-10-integration
git remote remove coinbase-archive
```

B.2 Resolve overlapping files (expected conflicts):
- `pyproject.toml` — manual merge: keep market-connector's structure, fold in coinbase's deps as `[project.optional-dependencies] coinbase = [...]`, drop coinbase's hatch-version + entry-points if any.
- `pixi.toml` — manual merge: fold coinbase deps into a `[feature.coinbase]` block, add `coinbase` env (and `coinbase-ci`).
- `LICENSE`, `README.md`, `.gitignore`, `.release-please-config.json`, `.release-please-manifest.json`, `.github/workflows/*` — keep market-connector's; discard coinbase's duplicates.
- `tests/conftest.py` — merge fixtures (root conftest stays the same; coinbase fixtures move under `tests/exchanges/coinbase/conftest.py`).

B.3 Update imports across the rewritten coinbase files:
- `from coinbase_connector.X` → `from market_connector.exchanges.coinbase.X`
  - All 18 module files + 14 test files (per file inventory above).
  - Use `Grep` + `Edit` (or a one-shot `sed -i` via `git mv`-style script).

B.4 Update `market_connector/exchanges/coinbase/__init__.py` to re-export the public surface:
```python
from market_connector.exchanges.coinbase.coinbase_gateway import CoinbaseGateway
from market_connector.exchanges.coinbase.config import CoinbaseConfig
__all__ = ["CoinbaseGateway", "CoinbaseConfig"]
```

B.5 Add `market_connector/exchanges/__init__.py`:
```python
"""Exchange-specific connectors implementing market_connector protocols.

Install with optional extras: `pip install hb-market-connector[coinbase]`.
"""
```

### Phase C — Build/CI integration on hb-market-connector (≈1 hour)

C.1 `pyproject.toml`:
- Add `[project.optional-dependencies] coinbase = ["pyjwt>=2.8", "cryptography>=41"]`.
- Extend `[tool.coverage.run] source_pkgs` to include `market_connector.exchanges.coinbase`.
- Extend `[tool.ruff] include` / `[tool.mypy] files` if scoped explicitly.
- Remove `[tool.hatch.metadata] allow-direct-references` (no longer needed; deps are now PyPI-pinned, not git URLs).

C.2 `pixi.toml`:
- Add `[feature.coinbase.dependencies]` with conda-forge equivalents (`pyjwt`, `cryptography`).
- Add `coinbase = { features = ["coinbase", "dev"], solve-group = "default" }` env.
- Extend `[feature.ci.dependencies]` to include `pyjwt`, `cryptography` so CI runs the coinbase tests.
- Update `[tasks]`: extend `lint`/`format`/`test` paths to include `market_connector/exchanges` and `tests/exchanges`.

C.3 CI workflow:
- Existing `.github/workflows/ci.yml` (ci-framework reusable) auto-discovers via pixi. Verify by running locally.
- No new workflows needed; matrix already covers 3.10/3.11/3.12.
- Confirm `enable-hygiene-check: true` still passes (the new exchanges/ dir gets scanned).

C.4 Local verification before pushing:
```
pixi install
pixi run -e ci lint
pixi run -e ci format-check
pixi run -e ci typecheck
pixi run -e ci test
pixi run -e coinbase test                   # exchange-specific run
```

### Phase D — PR + review on hb-market-connector (≈1 day for review)

D.1 Open PR `feat/exchanges-coinbase` → `development` with title:
`feat(exchanges): add coinbase exchange connector (merge from hb-coinbase-connector)`

D.2 PR description includes:
- Link to this migration plan
- Link to closed PR #1 on hb-coinbase-connector for archeology
- Summary of the boundary-correction rationale
- Test count delta (expected ~+150 tests)
- Confirmation that all CI is green (Python 3.10/3.11/3.12, Quality, Audit, security scans)

D.3 Once approved + merged:
- Tag a release on hb-market-connector: minor version bump (e.g., `v0.2.0`) since this is a feature addition, not a breaking change.
- Update `MementoRC/hb-market-connector` README to advertise `[coinbase]` extra.

### Phase E — Retire hb-coinbase-connector repo (≈30 min)

E.1 On `MementoRC/hb-coinbase-connector`:
- Close PR #1 with comment linking to the merge PR on hb-market-connector and to this plan.
- Tag `archive/pre-merge` (final state of feat/phase-10-integration).
- Add a notice to README.md: "**This repository has been merged into [hb-market-connector](https://github.com/MementoRC/hb-market-connector) as `market_connector.exchanges.coinbase`. Install with `pip install hb-market-connector[coinbase]`. This repo is archived for historical reference.**"
- Archive the repo via GitHub Settings → Archive.

E.2 Do NOT delete the repo (preserves issue history, CodeQL alerts, prior CI runs).

### Phase F — Update parent hummingbot repo (≈30 min)

Branch on `MementoRC/hummingbot`: `chore/retire-coinbase-connector-submodule` off `ci-base`.

F.1 Remove submodule:
```
git submodule deinit -f sub-packages/coinbase-connector
git rm -f sub-packages/coinbase-connector
rm -rf .git/modules/sub-packages/coinbase-connector
```

F.2 `.gitmodules`: ensure the `[submodule "sub-packages/coinbase-connector"]` block is gone.

F.3 Hummingbot code audit completed (Task 3): Zero imports of `coinbase_connector` package found outside sub-packages/coinbase-connector/ itself. Candles and rate-oracle reference the *name* only, not the package. Status: CLEAN.

F.4 Branch-tracking audit completed (Task 2):
   - custom_git_setup/configs/branch-tracking.yaml: NO references to hb-coinbase-connector found.
   - custom_git_setup/ directory scan: only test_history.json references (not config).
   - hb-market-connector: No special config in branch-tracking.yaml (will be added post-integration).
   - Action: No cleanup needed in branch-tracking.yaml before Phase F execution.

F.5 Commit with message: `chore(submodule): retire hb-coinbase-connector — merged into hb-market-connector`. GPG-sign per project rule.

F.6 Open PR on hummingbot — `chore/retire-coinbase-connector-submodule` → `ci-base`. After merge, the next bleeding-edge rebuild picks up the change automatically.

## 5. Validation Checklist

Pre-merge on hb-market-connector PR:
- [ ] All ci-framework jobs green (Quality, Audit, CodeQL, Semgrep, Secrets, Scorecard, Hygiene, Detect Changes)
- [ ] Test Python 3.10
- [ ] Test Python 3.11
- [ ] Test Python 3.12
- [ ] Coverage >= existing market-connector baseline (no regression)
- [ ] `pixi run -e coinbase test` runs the coinbase suite cleanly
- [ ] `pixi run -e ci test` runs everything including coinbase
- [ ] `pip install hb-market-connector` (no extras) does NOT pull pyjwt/cryptography
- [ ] `pip install hb-market-connector[coinbase]` DOES pull pyjwt/cryptography
- [ ] Importing `from market_connector.exchanges.coinbase import CoinbaseGateway` works
- [ ] No leftover `coinbase_connector` import strings (grep returns 0)

Post-merge:
- [ ] Tag + release on hb-market-connector
- [ ] PR #1 on hb-coinbase-connector closed with merge link
- [ ] hb-coinbase-connector repo archived
- [ ] hummingbot ci-base PR merged (submodule removed)
- [ ] Bleeding-edge rebuild succeeds on next cron run

## 6. Risk Register

Risk: filter-repo path collision (file already exists in market-connector)
  Likelihood: Low | Impact: Medium
  Mitigation: Pre-check; only LICENSE/README/pyproject overlap, all handled in B.2

Risk: Lost git-blame attribution
  Likelihood: Low (with A1) / High (with A3) | Impact: Low
  Mitigation: Use A1 strategy (filter-repo)

Risk: Test ordering / fixture collisions when suites combine
  Likelihood: Medium | Impact: Medium
  Mitigation: Run `pixi run -e ci test --collect-only` first; verify no duplicate test IDs

Risk: Pixi solver conflict between market-connector deps and coinbase JWT deps
  Likelihood: Low | Impact: Low
  Mitigation: Both are pure-python, no system-level conflicts expected

Risk: Future hummingbot-side imports break (none currently)
  Likelihood: Low | Impact: High
  Mitigation: F.3 grep audit before submodule removal (COMPLETED)

Risk: Someone has uncommitted work on the coinbase repo
  Likelihood: User-known | Impact: High
  Mitigation: User confirms before Phase A.2 archive tag

Risk: CodeQL / Semgrep raise NEW alerts on the merged code
  Likelihood: Medium | Impact: Low
  Mitigation: Address in follow-up commit if any; the original PR #1 already had the protocols.py:38 false-positive

## 7. Rollback

Phase B–C reversible: revert the merge commit on hb-market-connector, drop the branch.

Phase D reversible until PR merges: close the PR.

Phase E reversible: un-archive the repo, reopen PR #1.

Phase F reversible: revert the submodule-removal commit before bleeding-edge rebuild picks it up. After bleeding-edge rebuild, requires bleeding-edge fix-forward (re-add submodule pointing to archive tag).

Total point of no return: Phase E (archive) + Phase F merge to ci-base. Up until then, every step is revertible.

## 8. Effort Estimate

| Phase | Effort | Can parallelize? |
|-------|--------|------------------|
| A. Pre-migration audit | 30 min | — |
| B. Code migration | 2 hours | No (history rewrite + manual merges sequential) |
| C. Build/CI integration | 1 hour | After B |
| D. PR + review | 1 day calendar (~1 hour active) | — |
| E. Retire hb-coinbase-connector | 30 min | After D |
| F. Update hummingbot submodule | 30 min | Parallel with E |

**Total active work**: ~5–6 hours over 1–2 days.

## 9. Locked Decisions

1. **History strategy**: git filter-repo (install via `pixi global install git-filter-repo` or `pip install git-filter-repo`)
2. **Version bump**: minor bump on hb-market-connector (e.g., 0.1.x → 0.2.0)
3. **Optional extras naming**: `[coinbase]` (matches dir, future binance becomes `[binance]`)
4. **Tests location**: `tests/exchanges/coinbase/` mirroring source
5. **Archival**: archive `hb-coinbase-connector` repo (do not delete)
6. **Branch tracking**: No cleanup required — audit found zero references in branch-tracking.yaml to either hb-coinbase-connector or hb-market-connector.

## 10. Status: APPROVED — Proceeding to Phase A

Execution plan is locked. Proceed Phase A → F sequentially. No user input required.
