# Trailing Stop Manager

A utility that provides dynamic trailing stop management based on PNL levels.

## Overview

![Trailing Stop Behavior vs. PnL](assets/trailing_stop_basic.svg)

The trailing stop percentage increases with PNL:
- Starts at base trailing stop (1%)
- Increases linearly with slope defined by damping factor (0.9)
- Capped at maximum trailing stop (5%)

## Usage

```python
class TrailingStopManager:
    def __init__(
        self,
        trailing_stop_config: LadderedTrailingStop,
        get_trigger_pnl: Callable[[], Optional[Decimal]],
        set_trigger_pnl: Callable[[Optional[Decimal]], None],
        damping_factor: Decimal = Decimal("0.9"),
        max_trailing_pct: Decimal = Decimal("0.05"),
    ):