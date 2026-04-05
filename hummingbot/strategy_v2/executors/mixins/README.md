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

## Future Mixins (Not Yet Extracted)

These patterns were identified as duplicated but not yet extracted:

| Mixin | Duplicated In | Lines per executor | Notes |
|-------|---------------|-------------------|-------|
| **TripleBarrierMixin** | Position, DCA, Grid, Progressive | ~60 | SL/TP/TL/trailing dispatch — most complex extraction |
| **OrderTrackingMixin** | Position, DCA, Grid, Progressive | ~25 | `update_tracked_orders_with_order_id()` + `_failed_orders` |
| **PNLCalculatorMixin** | Position, DCA, Progressive | ~30 | `trade_pnl - fees` pattern (Grid uses different model) |
| **ActivationBoundsMixin** | Position, DCA, Grid, Progressive | ~15 | `_is_within_activation_bounds()` with BUY/SELL + MAKER/TAKER |

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
