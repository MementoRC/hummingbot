from decimal import Decimal
from typing import Callable

from hummingbot.strategy_v2.executors.progressive_executor.data_types import LadderedTrailingStop


class TrailingStopManager:
    """
    Controller class for managing trailing stops.
    Can be used by any executor that needs trailing stop functionality.

    Trailing stops are activated when the net PnL percentage exceeds the activation threshold.
    When the net PnL percentage drops below the trigger threshold, the trailing stop is triggered.
    The trailing stop can be configured to take profits at certain PnL levels.
    """

    def __init__(
            self,
            trailing_stop_config: LadderedTrailingStop,
            get_trigger_pnl: Callable[[], Decimal | None],
            set_trigger_pnl: Callable[[Decimal | None], None],
            damping_factor: Decimal = Decimal("0.9"),
            max_trailing_pct: Decimal = Decimal("0.05"),
    ):
        self._config = trailing_stop_config
        self._get_trigger_pnl = get_trigger_pnl
        self._set_trigger_pnl = set_trigger_pnl
        self._damping_factor = damping_factor
        self._max_trailing_pct = max_trailing_pct

    def update(
            self,
            net_pnl_pct: Decimal,
            current_amount: Decimal,
            on_close_position: Callable,
            on_partial_close: Callable,
    ) -> None:
        """
        Updates trailing stop state and executes callbacks when needed.

        :param net_pnl_pct: The net PnL percentage.
        :param current_amount: The current amount of the asset.
        :param on_close_position: Callback to close the position.
        :param on_partial_close: Callback to partially close the
        """
        assert current_amount > 0, f"Current amount must be positive: {current_amount} <= 0"

        trailing_pct = self._calculate_trailing_percentage(net_pnl_pct)
        trigger_pnl = self._get_trigger_pnl()

        if not trigger_pnl:
            if net_pnl_pct >= self._config.activation_pnl_pct:
                self._set_trigger_pnl(net_pnl_pct - trailing_pct)
            return

        if (updated_trigger := net_pnl_pct - trailing_pct) > trigger_pnl:
            self._set_trigger_pnl(updated_trigger)
            return

        if net_pnl_pct <= trigger_pnl:
            self._handle_stop_trigger(net_pnl_pct, current_amount, on_close_position, on_partial_close)
            self._set_trigger_pnl(net_pnl_pct - trailing_pct)

    def _calculate_trailing_percentage(self, net_pnl_pct: Decimal) -> Decimal:
        """
        Calculate the trailing percentage based on the net PnL percentage.
        At base, use config trailing_pct.
        As PNL grows, add a fraction (damping_factor) of the excess PNL.

        :param net_pnl_pct: The net PnL percentage.
        :return: The trailing percentage.
        """
        base_trailing = self._config.trailing_pct
        excess_pnl = max(net_pnl_pct - base_trailing, Decimal("0"))
        extra_trailing = excess_pnl * self._damping_factor

        return min(
            base_trailing + extra_trailing,
            self._max_trailing_pct
        )

    def _handle_stop_trigger(
            self,
            net_pnl_pct: Decimal,
            current_amount: Decimal,
            on_close_position: Callable,
            on_partial_close: Callable,
    ) -> None:
        closest_take_profit = max(
            filter(lambda x: x[0] <= net_pnl_pct, self._config.take_profit_table),
            key=lambda x: x[0],
            default=(Decimal("0"), Decimal("1"))
        )
        close_ratio = closest_take_profit[1]

        if close_ratio == Decimal("1"):
            on_close_position()
        else:
            on_partial_close(current_amount * close_ratio)

    def _find_closest_take_profit(self, net_pnl_pct: Decimal) -> tuple[Decimal, Decimal]:
        return max(
            filter(lambda x: x[0] <= net_pnl_pct, self._config.take_profit_table),
            key=lambda x: x[0],
            default=(Decimal("0"), Decimal("1"))
        )