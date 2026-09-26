# Executor Mixins

Shared reusable behaviors extracted from the executor codebase to eliminate
duplicated code and standardize patterns across all executors.

## Available Mixins

### RetryMixin (`retry.py`)
Tracks retry count and evaluates max-retries threshold.

**Applied to:** PositionExecutor, DCAExecutor, GridExecutor, TWAPExecutor

**Usage:**
```python
class MyExecutor(RetryMixin, ExecutorBase):
    def __init__(self, ..., max_retries=10):
        super().__init__(...)
        self.init_retry(max_retries)

    async def control_task(self):
        # ... your logic ...
        self.evaluate_max_retries()  # call at end of each tick

    def process_order_failed_event(self, ...):
        self.increment_retries("order failed")
```

**Methods:**
- `init_retry(max_retries)` — initialize retry state
- `increment_retries(reason)` — bump counter
- `reset_retries()` — reset to 0
- `evaluate_max_retries()` — stop if `current > max` (auto-detects close_execution_by vs stop)
- `current_retries` / `max_retries` — properties with getter/setter

### BalanceValidationMixin (`balance_validation.py`)
Validates sufficient balance before executor start.

**Applied to:** PositionExecutor, GridExecutor, TWAPExecutor
**Not applied to:** DCAExecutor (validates multiple candidates per level)

**Usage:**
```python
class MyExecutor(BalanceValidationMixin, ExecutorBase):
    def _create_validation_order_candidate(self):
        return OrderCandidate(
            trading_pair=self.config.trading_pair,
            is_maker=False,
            order_type=OrderType.MARKET,
            order_side=self.config.side,
            amount=self.config.amount,
            price=self.get_price(),
        )
```

**Methods:**
- `validate_sufficient_balance()` — builds candidate, checks via budget checker, stops if insufficient
- `_create_validation_order_candidate()` — **template method** — override to provide the order candidate

### ShutdownMixin (`shutdown.py`)
Template for standardized shutdown process.

**Status:** Created but NOT applied to any executor. Each executor's
`control_shutdown_process()` has significant unique branching that
doesn't cleanly fit the template pattern.

**Available for:** New executors with simpler shutdown requirements.

**Template methods to override:**
- `_get_open_filled_amount()` → Decimal
- `_get_close_filled_amount()` → Decimal
- `_get_min_order_size()` → Decimal
- `_has_pending_close_order()` → bool
- `_place_shutdown_close_order()` → None
- `_update_orders_with_error_handler()` → None

### OrderTrackingMixin (`order_tracking.py`)
Provides `update_tracked_orders_with_order_id()` which fetches the
InFlightOrder from the connector and assigns it to the matching
TrackedOrder. Also manages `_failed_orders` list.

**Applied to:** DCAExecutor, TWAPExecutor
**Not applied to:** PositionExecutor (checks specific named fields, not a list),
GridExecutor (iterates nested grid level structure with `update_grid_levels()` call)

**Usage:**
```python
class MyExecutor(OrderTrackingMixin, ExecutorBase):
    def __init__(self, ...):
        super().__init__(...)
        self.init_order_tracking()

    def _get_trackable_orders(self):
        return self._open_orders + self._close_orders
```

**Methods:**
- `init_order_tracking()` — initialize `_failed_orders` list
- `update_tracked_orders_with_order_id(order_id)` — fetch InFlightOrder and assign
- `_get_trackable_orders()` — **template method** — override to return orders to search

### ActivationBoundsMixin (`activation_bounds.py`)
Checks whether the current market price is within configured activation
bounds of the target order price — improving capital efficiency.

**Applied to:** PositionExecutor
**Not applied to:** DCAExecutor (different signature, has MAKER/TAKER mode split,
uses different operators and `close_price` instead of `mid_price`),
GridExecutor (uses single Decimal `activation_bounds`, not a list — completely
different distance-based filtering pattern),
TWAPExecutor (no activation bounds)

**Usage:**
```python
class MyExecutor(ActivationBoundsMixin, ExecutorBase):
    def control_open_order(self):
        if self._is_within_activation_bounds(
            self.config.entry_price,
            self.config.side,
            self.config.triple_barrier_config.open_order_type,
        ):
            self.place_open_order()
```

**Methods:**
- `_is_within_activation_bounds(order_price, side, order_type)` — checks limit (one-sided) vs market (two-sided range)

### TrailingStopMixin (`trailing_stop.py`)
Provides the trailing stop ratchet algorithm: activate when PNL exceeds
threshold, ratchet trigger upward as PNL rises, fire when PNL drops below.

**Applied to:** PositionExecutor, DCAExecutor, GridExecutor
**Not applied to:** TWAPExecutor (no trailing stop support)

**Usage:**
```python
class MyExecutor(TrailingStopMixin, ExecutorBase):
    def __init__(self, ...):
        super().__init__(...)
        self.init_trailing_stop()

    def control_trailing_stop(self):
        if self.evaluate_trailing_stop():
            self.place_close_order_and_cancel_open_orders(
                close_type=CloseType.TRAILING_STOP)

    def _get_trailing_stop_pnl_pct(self):
        return self.get_net_pnl_pct()

    def _get_trailing_stop_config(self):
        return self.config.trailing_stop  # object with .activation_price, .trailing_delta
```

**Methods:**
- `init_trailing_stop()` — initialize trailing stop state
- `evaluate_trailing_stop()` — returns True if trailing stop should fire
- `_get_trailing_stop_pnl_pct()` — **template method** — the PNL metric to track
- `_get_trailing_stop_config()` — **template method** — config with activation_price and trailing_delta

### PNLCalculatorMixin (`pnl_calculator.py`)
Provides `trade_pnl_pct`, `trade_pnl_quote`, `get_net_pnl_quote()`,
`get_cum_fees_quote()`, `get_net_pnl_pct()` using the trade_pnl - fees pattern.

**Applied to:** DCAExecutor, TWAPExecutor
**Not applied to:** PositionExecutor (has FAILED/POSITION_HOLD guard in trade_pnl_pct),
GridExecutor (realized/unrealized split model), XEMM/Arbitrage (cash flow difference model)

**Usage:**
```python
class MyExecutor(PNLCalculatorMixin, ExecutorBase):
    def _get_entry_price(self):
        return self.current_position_average_price

    def _get_close_price(self):
        return self.close_price

    def _get_open_filled_amount_quote(self):
        return self.open_filled_amount_quote

    def _get_trade_side(self):
        return self.config.side

    def _get_cum_fees_from_orders(self):
        return sum(o.cum_fees_quote for o in self._all_orders if o)
```

**Provided properties/methods:**
- `trade_pnl_pct` (property) — pure PNL percentage without fees
- `trade_pnl_quote` (property) — pure PNL in quote currency
- `get_net_pnl_quote()` — trade_pnl_quote minus fees
- `get_cum_fees_quote()` — cumulative fees
- `get_net_pnl_pct()` — net PNL percentage

**Template methods:**
- `_get_entry_price()` — entry/average price
- `_get_close_price()` — close/current market price
- `_get_open_filled_amount_quote()` — filled amount in quote
- `_get_trade_side()` — TradeType.BUY or SELL
- `_get_cum_fees_from_orders()` — sum of fees from orders

## Future Mixins (Not Yet Extracted)

These patterns were identified as duplicated but not yet extracted:

| Mixin | Duplicated In | Lines per executor | Notes |
|-------|---------------|-------------------|-------|
| **TripleBarrierMixin** | Position, DCA, Grid, Progressive | ~60 | SL/TP/TL dispatch — most complex extraction |

## Design Principles

1. **Template methods over inheritance** — Mixins define hooks (`_create_*`, `_get_*`, `_has_*`) that executors override, rather than trying to handle all cases in the mixin itself.

2. **Only extract what matches 90%+** — If an executor's implementation differs significantly (e.g., DCA's multi-candidate balance validation, Grid's POSITION_HOLD shutdown), leave it as-is rather than forcing it into a mixin.

3. **MRO order matters** — Mixins go BEFORE ExecutorBase in the class definition:
   ```python
   class MyExecutor(RetryMixin, BalanceValidationMixin, ExecutorBase):
   ```

4. **No circular imports** — Mixins import only from `models/` and `core/`, never from executor implementations.

5. **Works with ExecutorFactory** — The `@ExecutorFactory.register(Config)` decorator goes on the final class, not the mixins.

## How to Add a New Mixin

1. Create `hummingbot/strategy_v2/executors/mixins/my_mixin.py`
2. Define the mixin class with template methods (raise NotImplementedError)
3. Add tests in `test/hummingbot/strategy_v2/executors/mixins/test_my_mixin.py`
4. Apply to executors where the pattern matches cleanly
5. Update this README
