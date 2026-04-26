# hb-coinbase-connector Archive Manifest

**Archived**: 2026-04-25
**Source repo**: MementoRC/hb-coinbase-connector
**Source branch**: feat/phase-10-integration
**Archive tag**: archive/pre-merge-into-market-connector
**HEAD commit**: cd21653d036b1ef566556ffced81a8a47abc4acc
**Total commits**: 40
**Time span**: 2026-04-24 15:20:40 (-0600) to 2026-04-25 15:47:00 (-0500)
**Status of PR #1**: closed (CI green: tests 3.10/3.11/3.12, Quality, Audit, Build, security all pass; CodeQL neutral due to protocols.py:38 false-positive)

This file preserves a one-line summary of every commit on `feat/phase-10-integration` before the migration to `hb-market-connector` rewrites paths via `git filter-repo`. Use the original SHAs (left column) to inspect any commit on the archive tag.

---

## Commit Manifest (newest first)

| SHA (short) | Date | Subject |
|---|---|---|
| cd21653d | 2026-04-25 | fix(build): allow direct references in hatch metadata for git URL dep |
| 726c6453 | 2026-04-25 | fix(ci): remove redundant security.yml, replace path dep with git URL, lower python floor |
| 98fe237d | 2026-04-25 | test(phase-10): raise coverage to 98.8% — add missing branch tests |
| bf46c5a1 | 2026-04-25 | fix(tools): import order in fixture_recorder (I001) |
| a67ca52a | 2026-04-25 | feat(tools): add Coinbase fixture recorder CLI |
| 3193b116 | 2026-04-25 | style: apply ruff lint+format cleanup across modules |
| 2c42b448 | 2026-04-25 | test(contract): inherit hb-market-connector contract suite for CoinbaseGateway |
| 4a3e50b8 | 2026-04-25 | feat(api): expose CoinbaseGateway and CoinbaseConfig as public package API |
| 0c84f667 | 2026-04-25 | feat(gateway): compose CoinbaseGateway from mixins with auth/rest/ws orchestration |
| 4cdf36da | 2026-04-25 | feat(mixins): add SubscriptionsMixin (subscribe_orderbook, subscribe_trades) |
| 30de41e7 | 2026-04-25 | feat(mixins): add OrdersMixin (place, cancel, get_open_orders) |
| 5d192e12 | 2026-04-25 | feat(mixins): add MarketDataMixin (orderbook, mid_price, candles) |
| 87f1f943 | 2026-04-25 | feat(mixins): add AccountsMixin (get_balance) |
| 21300d0c | 2026-04-25 | fix(typecheck): resolve mypy errors in auth.py and converters.py |
| 2559bf3d | 2026-04-25 | feat(protocols): define mixin Protocol types for gateway composition |
| cdb4eb23 | 2026-04-25 | refactor(converters): fix lint violations (F401 unused import, I001 import order) |
| 35820736 | 2026-04-25 | feat(converters): add MarketTrade → TradeEvent and Candle → OHLCV list converters |
| 557d8c07 | 2026-04-25 | feat(converters): add orderbook snapshot and incremental update converters |
| cfc250ad | 2026-04-25 | feat(converters): add Coinbase Order → OpenOrder converter with order-type extraction |
| 443d2963 | 2026-04-25 | feat(converters): add Account → available balance (Decimal) converter |
| 60878e5e | 2026-04-25 | feat(converters): add pair passthrough converters (to/from_exchange_pair) |
| 750fb1ec | 2026-04-25 | refactor(schemas): fix lint violations (UP045, N815, B017, E501) |
| 9b174fd3 | 2026-04-25 | feat(schemas): add WS message models |
| fed562d5 | 2026-04-25 | feat(schemas): add REST response models with fixture validation |
| f82b33c8 | 2026-04-25 | feat(schemas): add Coinbase enum types |
| 9bf85f4e | 2026-04-25 | fix(test): rename test_endpoint_is_Endpoint_type to lowercase (N802) |
| c436f800 | 2026-04-25 | feat(config): add CoinbaseConfig with sandbox URL switching |
| b6021ab9 | 2026-04-25 | feat(endpoints): add ENDPOINT_REGISTRY with per-endpoint rate limits |
| 5a4e641b | 2026-04-25 | test(auth): add ES256 round-trip signature verification |
| 2ae2a9ee | 2026-04-25 | fix(transport): enforce https:// in CoinbaseRestClient base_url |
| e4e43ebf | 2026-04-25 | fix(auth): add aud=cdp claim to JWT for Coinbase CDP audience binding |
| de0b73e7 | 2026-04-25 | fix(auth): apply ruff auto-fixes |
| 93ba0bd6 | 2026-04-25 | feat(transport): step 5 — CoinbaseRestClient for context-aware signing |
| b30743d3 | 2026-04-25 | feat(auth): step 4 — coinbase_auth() AuthCallable factory with JWT+HMAC dispatch |
| a41d2592 | 2026-04-25 | feat(auth): step 3 — HMAC-SHA256 signer |
| 19083260 | 2026-04-25 | feat(auth): step 2 — JWT builder (ES256, REST+WS variants) |
| 797246f9 | 2026-04-25 | feat(auth): step 1 — PEM normalization for JWT key material |
| 3b7c9d78 | 2026-04-24 | chore: scaffold hb-coinbase-connector package structure |
| bf9f0a9d | 2026-04-24 | feat: initial scaffold for hb-coinbase-connector |
| 6b7d370c | 2026-04-24 | Initial commit |

## Recovery instructions

To inspect the archive after migration:

```
git clone https://github.com/MementoRC/hb-coinbase-connector
cd hb-coinbase-connector
git checkout archive/pre-merge-into-market-connector
git log --oneline
```

To re-extract any single file at any historical SHA:

```
git show <sha>:<original-path>
# original-path uses pre-rewrite paths (coinbase_connector/..., tests/...)
```

After migration to hb-market-connector, the same content lives at:
- `coinbase_connector/X` → `market_connector/exchanges/coinbase/X`
- `tests/X` → `tests/exchanges/coinbase/X`
