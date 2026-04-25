# hb-coinbase-connector Design Spec

**Date:** 2026-04-24
**Status:** Draft
**Repository:** MementoRC/hb-coinbase-connector
**Submodule:** sub-packages/coinbase-connector

## Overview

Reference implementation of the `ExchangeGateway` protocol (from hb-market-connector) for the Coinbase Advanced Trade API. This connector demonstrates how exchange-specific code maps onto the gateway framework's transport layer, protocol contracts, and testing infrastructure.

## Architecture

**Approach:** Mixin inheritance with Protocol-typed `self` and gateway transport injection.

`CoinbaseGateway` composes four domain-aligned mixins via multiple inheritance. Each mixin declares its dependencies through Protocol-typed `self` parameters (e.g., `self: HasRest & HasAuth`), eliminating the SuperCalls shim pattern used in the archived connector. The gateway class is a thin composition root — it wires transport, auth, and config but contains no business logic.

**Data flow:**

```
Coinbase API (JSON) → Pydantic v2 schema models → pure converter functions → gateway primitives
```

The converter layer is the single boundary between exchange-specific and exchange-neutral types. Mixins call converters — they never touch raw JSON or construct primitives directly.

**Key architectural properties:**
- Structural subtyping: `CoinbaseGateway` satisfies `ExchangeGateway` protocol without inheriting from it
- Transport injection: `RestConnectorBase` and `WsConnectorBase` from the gateway framework provide retry, reconnection, and rate limiting
- Auth as callable: `AuthCallable = Callable[[dict], Awaitable[dict]]` — pure transformation, no transport awareness
- Each mixin is independently testable via mock transport from the gateway framework

## Package Structure

```
coinbase_connector/
├── __init__.py                     # Public API: CoinbaseGateway, CoinbaseConfig
├── coinbase_gateway.py             # Composition root — wires mixins + transport + lifecycle
├── config.py                       # CoinbaseConfig (Pydantic v2: api_key, secret, sandbox flag)
├── auth.py                         # coinbase_auth() → AuthCallable (JWT primary + HMAC fallback)
├── endpoints.py                    # URL constants, endpoint registry, rate limit specs
├── converters.py                   # Pure functions: schema models → gateway primitives
│
├── schemas/
│   ├── __init__.py
│   ├── rest.py                     # Pydantic v2 models for all REST responses
│   ├── ws.py                       # Pydantic v2 models for all WS message types
│   └── enums.py                    # Exchange-side enums (OrderStatus, ProductType, etc.)
│
├── mixins/
│   ├── __init__.py
│   ├── protocols.py                # Protocol types for self-typing across mixins
│   ├── orders.py                   # OrdersMixin: place, cancel, get_open_orders
│   ├── accounts.py                 # AccountsMixin: get_balance
│   ├── market_data.py              # MarketDataMixin: orderbook, candles, mid_price
│   └── subscriptions.py           # SubscriptionsMixin: subscribe_orderbook, subscribe_trades
│
├── tools/
│   ├── __init__.py
│   └── fixture_recorder.py        # Automated API response capture + sanitization
│
└── tests/
    ├── __init__.py
    ├── fixtures/                    # Captured JSON responses (REST + WS)
    │   ├── rest/                    # products.json, orders.json, accounts.json, ...
    │   └── ws/                     # level2.json, market_trades.json, user.json, ...
    ├── test_schemas.py             # Tier 1: validate models against fixtures
    ├── test_converters.py          # Tier 2: schema → primitive pure function tests
    ├── test_auth.py                # Auth signing verification
    ├── test_orders_mixin.py        # Tier 3: mock transport, order flow
    ├── test_accounts_mixin.py
    ├── test_market_data_mixin.py
    ├── test_subscriptions_mixin.py
    └── test_contract.py            # Tier 4: GatewayContractTestBase subclass
```

## Auth Module

**File:** `auth.py`

Exports a factory function:

```python
def coinbase_auth(api_key: str, secret_key: str) -> AuthCallable:
```

**JWT flow (primary):** Extracted from the existing `CoinbaseAdvancedTradeAuth` in hummingbot.
- `_normalize_pem(secret_key)` — handles raw base64, single-line PEM, multi-line PEM → valid EC PEM
- ES256 signing: `sub=api_key`, `iss="cdp"`, `kid=api_key`, random `nonce`, 120s expiry
- REST: `uri` claim = `"METHOD api.coinbase.com/path"`, returns `{"Authorization": "Bearer <jwt>"}`
- WS: no `uri` claim, returns `{"jwt": token}` for injection into subscribe payload

**HMAC fallback:** Activated if PEM normalization fails.
- `HMAC-SHA256(secret, timestamp + method + path + body)`
- REST: returns `{"CB-ACCESS-KEY": ..., "CB-ACCESS-SIGN": ..., "CB-ACCESS-TIMESTAMP": ...}`
- WS: returns `{"api_key": ..., "signature": ..., "timestamp": ...}`

**WS auth caveat:** WS subscribe payloads inject auth fields into the message body (not HTTP headers). `SubscriptionsMixin` calls auth separately and merges the result into the subscribe JSON.

**Dependencies:** `pyjwt`, `cryptography` (already in hummingbot's environment).

**Reference:** `coinbase-advanced-py` SDK source used to validate signing logic — no runtime dependency.

## Schemas

**File:** `schemas/enums.py`, `schemas/rest.py`, `schemas/ws.py`

Full Pydantic v2 coverage (`BaseModel` with `ConfigDict(frozen=True)`) for all consumed API types. Validated against `coinbase-advanced-py` source and captured fixtures.

### Enums (`schemas/enums.py`)

- `CoinbaseOrderStatus` — OPEN, FILLED, CANCELLED, PENDING, FAILED
- `CoinbaseOrderType` — MARKET, LIMIT, LIMIT_MAKER, STOP_LIMIT
- `CoinbaseSide` — BUY, SELL
- `CoinbaseProductType` — SPOT, FUTURE
- `CoinbaseGranularity` — ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, SIX_HOUR, ONE_DAY
- `CoinbaseWsChannel` — level2, market_trades, user, candles, ticker, status

### REST Response Models (`schemas/rest.py`)

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `ProductResponse` / `Product` | Trading pair metadata | product_id, base/quote currency, price/size increments |
| `AccountResponse` / `Account` | Balance info | uuid, currency, available_balance, hold |
| `OrderResponse` / `Order` | Order lifecycle | order_id, client_order_id, status, side, type, filled_size, average_filled_price |
| `FillResponse` / `Fill` | Trade fill details | trade_id, order_id, price, size, commission |
| `CandleResponse` / `Candle` | OHLCV data | start, low, high, open, close, volume |
| `OrderBookResponse` / `OrderBookLevel` | Book snapshot | bids/asks with price/size |
| `ServerTimeResponse` | Time sync | iso, epochSeconds |
| `TransactionSummaryResponse` | Fee tiers | fee_tier, total_volume |

Each response model has a top-level wrapper matching the exact Coinbase JSON envelope.

### WS Message Models (`schemas/ws.py`)

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `WsMessage` | Envelope | channel, client_id, timestamp, sequence_num, events |
| `Level2Event` / `Level2Update` | Orderbook diffs | type (snapshot/update), side, price_level, new_quantity |
| `MarketTradeEvent` / `MarketTrade` | Trade ticks | trade_id, side, price, size, time |
| `UserEvent` / `UserOrder` | Order updates | order_id, status, filled_size, avg_price |
| `CandleEvent` / `CandleUpdate` | Candle updates | start, low, high, open, close, volume |

## Converters

**File:** `converters.py`

Pure functions, no state, no I/O. Each takes a schema model and returns a gateway primitive:

| Function | Input | Output |
|----------|-------|--------|
| `to_open_order(order)` | `Order` | `OpenOrder` |
| `to_trade_event(trade, trading_pair)` | `MarketTrade` | `TradeEvent` |
| `to_orderbook_snapshot(book)` | `OrderBookResponse` | `OrderBookSnapshot` |
| `to_orderbook_update(event)` | `Level2Event` | `OrderBookUpdate` |
| `to_candle(candle)` | `Candle` | `list` — `[timestamp, open, high, low, close, volume]` matching gateway candle format |
| `to_balance(account)` | `Account` | `Decimal` |
| `to_exchange_pair(trading_pair)` | `str` | `str` (passthrough — Coinbase uses BTC-USD natively) |
| `from_exchange_pair(product_id)` | `str` | `str` (passthrough) |

The converter layer is the **only module** that imports both schema types and gateway primitives.

## Mixins

**File:** `mixins/protocols.py`, `mixins/orders.py`, `mixins/accounts.py`, `mixins/market_data.py`, `mixins/subscriptions.py`

### Mixin Protocols (`mixins/protocols.py`)

```python
class HasRest(Protocol):
    _rest: RestConnectorBase

class HasWs(Protocol):
    _ws: WsConnectorBase

class HasAuth(Protocol):
    _auth: AuthCallable

class HasEndpoints(Protocol):
    _endpoints: dict[str, Endpoint]  # gateway framework Endpoint type with rate limit fields

class HasConfig(Protocol):
    _config: CoinbaseConfig
```

Mixins combine these via intersection: `self: HasRest & HasAuth & HasEndpoints`.

### OrdersMixin (`mixins/orders.py`)

**Protocol methods:** `place_order`, `cancel_order`, `get_open_orders`

- `place_order`: Builds Coinbase order configuration (LIMIT → `limit_limit_gtc`, MARKET → `market_market_ioc`, LIMIT_MAKER → `limit_limit_fok`), POSTs to `/brokerage/orders`, parses `OrderResponse`, returns `client_order_id`
- `cancel_order`: POSTs to `/brokerage/orders/batch_cancel` with single order ID, returns success boolean
- `get_open_orders`: GETs `/brokerage/orders` with status=OPEN filter, converts via `to_open_order()`

### AccountsMixin (`mixins/accounts.py`)

**Protocol methods:** `get_balance`

- GETs `/brokerage/accounts`, finds account by currency, converts via `to_balance()`
- Short TTL cache (30s) on account list to avoid repeated calls when checking multiple currencies; invalidated on any balance-affecting operation (order placement/cancellation)

### MarketDataMixin (`mixins/market_data.py`)

**Protocol methods:** `get_orderbook`, `get_candles`, `get_mid_price`

- `get_orderbook`: GETs `/brokerage/product_book`, converts via `to_orderbook_snapshot()`
- `get_candles`: GETs `/products/{id}/candles` with granularity mapping, converts via `to_candle()`
- `get_mid_price`: Delegates to `get_orderbook()`, computes `(best_bid + best_ask) / 2`
- All public endpoints — no auth required

### SubscriptionsMixin (`mixins/subscriptions.py`)

**Protocol methods:** `subscribe_orderbook`, `subscribe_trades`

- `subscribe_orderbook`: Returns `AsyncContextManager` yielding `OrderBookUpdate` events. Subscribes to `level2` WS channel. First message treated as snapshot (hybrid init). Subsequent messages are deltas converted via `to_orderbook_update()` and delivered via callback. On reconnect: REST fallback via `get_orderbook()`. Sequence gap detection triggers re-sync. The context manager wraps a `Subscription` handle from `WsConnectorBase` — exiting the context cancels the subscription.
- `subscribe_trades`: Returns `AsyncContextManager` yielding `TradeEvent` events. Subscribes to `market_trades` WS channel. Each message converted via `to_trade_event()` and delivered via callback. Context manager lifecycle mirrors `subscribe_orderbook`.
- WS auth: injects auth fields into subscribe payload for authenticated channels

**Return type contract:** Both subscription methods return `AsyncContextManager` as required by the `MarketDataGateway` protocol. The context manager's `__aenter__` establishes the WS subscription; `__aexit__` cancels it. This maps directly to `WsConnectorBase.subscribe()` which returns a `Subscription` with a `cancel()` method.

**Cross-mixin dependency:** `subscribe_orderbook` needs `HasWs` (streaming) and access to `get_orderbook` (REST fallback on reconnect). Resolved by requiring `HasRest` on `self` — the composed `CoinbaseGateway` provides both via `MarketDataMixin`.

## CoinbaseGateway

**File:** `coinbase_gateway.py`

```python
class CoinbaseGateway(OrdersMixin, AccountsMixin, MarketDataMixin, SubscriptionsMixin):
    """Coinbase Advanced Trade gateway implementing ExchangeGateway protocol."""
```

**Composition root responsibilities:**
- Creates `RestConnectorBase(base_url, endpoints, auth, max_retries=3, retry_delay=1.0)`
- Creates `WsConnectorBase(ws_url, auth, heartbeat_interval=30, reconnect_delay=1.0, max_reconnect_delay=60)`
- `start()`: validates connectivity via `GET /brokerage/time`, connects WS
- `stop()`: disconnects WS (auto-cancels subscriptions), idempotent
- `ready` property: returns `self._started`
- Pre-start guard: all mixin methods check `self.ready`, raise `GatewayNotStartedError` if false

## CoinbaseConfig

**File:** `config.py`

```python
class CoinbaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: str
    secret_key: str
    sandbox: bool = False

    # Computed: base_url, ws_url (sandbox vs production)
```

**URLs:**
- Production REST: `https://api.coinbase.com/api/v3`
- Production WS: `wss://advanced-trade-ws.coinbase.com`
- Sandbox REST: `https://api-sandbox.coinbase.com/api/v3`
- Sandbox WS: `wss://advanced-trade-ws-sandbox.coinbase.com`

## Endpoints

**File:** `endpoints.py`

```python
# Uses the gateway framework's Endpoint type which embeds rate limit params per-endpoint
ENDPOINT_REGISTRY: dict[str, Endpoint] = {
    "server_time":    Endpoint(path="/brokerage/time", limit=10, window=1),
    "products":       Endpoint(path="/brokerage/market/products", limit=10, window=1),
    "product_book":   Endpoint(path="/brokerage/product_book", limit=10, window=1),
    "candles":        Endpoint(path="/brokerage/market/products/{product_id}/candles", limit=10, window=1),
    "accounts":       Endpoint(path="/brokerage/accounts", limit=30, window=1),
    "place_order":    Endpoint(path="/brokerage/orders", limit=30, window=1),
    "cancel_orders":  Endpoint(path="/brokerage/orders/batch_cancel", limit=30, window=1),
    "order_status":   Endpoint(path="/brokerage/orders/historical/{order_id}", limit=30, window=1),
    "order_fills":    Endpoint(path="/brokerage/orders/historical/fills", limit=30, window=1),
    "fee_summary":    Endpoint(path="/brokerage/transaction_summary", limit=30, window=1),
}
# Public endpoints: 10 req/s, Private endpoints: 30 req/s (embedded in Endpoint objects)
```

## Error Handling

**Error mapping (single `_handle_error()` utility function):**

| Coinbase Response | Gateway Exception |
|---|---|
| 401 Unauthorized | `AuthenticationError` |
| 429 Too Many Requests | `RateLimitError` |
| 400 + INVALID_ORDER / insufficient funds | `OrderRejectedError` |
| 404 + order not found | `OrderNotFoundError` |
| 503 / connection timeout / WS disconnect | `ExchangeUnavailableError` |

**Rate limiting:** Leverages `RestConnectorBase` built-in retry with exponential backoff. `RATE_LIMITS` dict annotates endpoint pool membership. On 429 after retries exhausted, `RateLimitError` propagates.

**WS resilience:** `WsConnectorBase` auto-reconnects with exponential backoff (max 60s). `SubscriptionsMixin` adds:
- Re-subscribe with fresh auth on reconnect
- Orderbook REST snapshot fallback on reconnect
- Sequence gap detection → REST re-sync

## Testing Strategy

### Tier 1 — Schema Tests (`test_schemas.py`)
Load fixture JSON → `Model.model_validate(fixture)` → assert field types/values. Catches API drift.

### Tier 2 — Converter Tests (`test_converters.py`)
Construct schema instances → call converter → assert gateway primitive fields. Pure functions, no I/O. Edge cases: empty orderbook, zero balance, partial fills, unknown status.

### Tier 3 — Mixin Tests (`test_*_mixin.py`)
Minimal test class inheriting single mixin + `MockRestClient`/`MockWsClient` from gateway framework. Register fixture responses → call mixin method → assert converter output. ~5-10 tests per mixin.

### Tier 4 — Contract Tests (`test_contract.py`)
Subclass `GatewayContractTestBase`, provide `gateway()` and `trading_pair()` fixtures. Inherits all protocol compliance tests (lifecycle, execution, market data). Mock transport — no real API calls.

### Fixture Recorder (`tools/fixture_recorder.py`)

CLI tool for automated fixture capture:

```bash
python -m coinbase_connector.tools.fixture_recorder \
    --api-key $KEY --secret $SECRET \
    --output tests/fixtures/ \
    --endpoints products,accounts,product_book,candles,server_time
```

**Flow:** Authenticate → hit each endpoint → sanitize (replace account UUIDs, API keys; preserve market data) → write JSON fixtures. For WS: subscribe to channels for ~5 seconds, capture snapshot + deltas.

**Rerunnable:** Diff against previous fixtures to detect API changes.

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Extract & adapt auth from in-tree connector | JWT/HMAC edge cases (PEM normalization, nonce) already solved |
| 2 | `coinbase-advanced-py` as reference only | No runtime dependency; validates schema correctness |
| 3 | Full Pydantic v2 schema coverage | Self-documenting, catches API drift, enables fixture-driven testing |
| 4 | Reimplement candles in-connector | Basis for comparison; future dependency inversion for hb-candles-feed |
| 5 | Hybrid orderbook init (WS snapshot + REST fallback) | Coinbase sends initial snapshot on subscribe; REST fallback for reconnection |
| 6 | Automated fixture recorder | Rerunnable capture; manual capture is one-time and tedious |
| 7 | 4 domain-aligned mixins | Proven granularity from archived connector; independently testable |
| 8 | Protocol flexibility | Revise gateway protocol when connector reveals better boundaries |
| 9 | Full scope (all 12 methods) | Archived connector provides complete reference; gateway contracts are well-defined |

## Future Work (Out of Scope)

- **BeautifulSoup API doc scraper:** Live validation of Pydantic models against Coinbase web documentation (archived connector had this — noted for future)
- **WS mechanics standardization:** Extract hybrid orderbook init pattern into reusable gateway framework component
- **hb-candles-feed dependency inversion:** Once connector candles are proven, hb-candles-feed could consume this connector instead of reimplementing
- **User channel (order status streaming):** WS `user` channel for real-time order updates — additive after initial implementation

## References

- **Gateway framework:** `sub-packages/market-connector/market_connector/` — protocols, primitives, transport, contract tests
- **Existing in-tree connector:** `hummingbot/connector/exchange/coinbase_advanced_trade/` — auth, API patterns
- **Archived connector:** `~/PycharmProjects/Archives/HummingbotWorktree/Feat_coinbase_advanced_trading/hummingbot/connector/exchange/coinbase_advanced_trade/` — mixin composition, Pydantic schemas, protocol-typed self
- **Candles adapter:** `sub-packages/candles-feed/candles_feed/adapters/coinbase_advanced_trade/` — REST/WS candle patterns
- **Protocol-typed self reference:** `dev/trailing_stop_strategy` branch — superior to NotImplementedError templates
