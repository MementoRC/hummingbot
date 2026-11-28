"""Shared PNL calculation for single-position executors."""

from __future__ import annotations

from decimal import Decimal

from hummingbot.core.data_type.common import TradeType


class PNLCalculatorMixin:
    """Mixin for PNL calculation using the trade_pnl - fees pattern.

    Applies to executors with a single entry/exit model (DCA, TWAP).
    Does NOT apply to Grid (realized/unrealized split model),
    XEMM/Arbitrage (cash flow difference model), or Position
    (has FAILED/POSITION_HOLD guard in trade_pnl_pct).

    Usage:
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
    """

    @property
    def trade_pnl_pct(self) -> Decimal:
        """Calculate the trade pnl (pure pnl without fees)."""
        entry = self._get_entry_price()
        close = self._get_close_price()
        if entry == Decimal("0"):
            return Decimal("0")
        if self._get_trade_side() == TradeType.BUY:
            return (close - entry) / entry
        else:
            return (entry - close) / entry

    @property
    def trade_pnl_quote(self) -> Decimal:
        """Calculate the trade pnl in quote asset."""
        return self.trade_pnl_pct * self._get_open_filled_amount_quote()

    def get_net_pnl_quote(self) -> Decimal:
        """Calculate the net pnl in quote asset."""
        return self.trade_pnl_quote - self.cum_fees_quote

    def get_cum_fees_quote(self) -> Decimal:
        """Calculate the cumulative fees in quote asset."""
        return self._get_cum_fees_from_orders()

    def get_net_pnl_pct(self) -> Decimal:
        """Calculate the net pnl percentage."""
        filled_quote = self._get_open_filled_amount_quote()
        if filled_quote == Decimal("0") or filled_quote <= Decimal("0"):
            return Decimal("0")
        return self.net_pnl_quote / filled_quote

    # Template methods — override in subclasses

    def _get_entry_price(self) -> Decimal:
        """Override: return the entry/average price for the position."""
        raise NotImplementedError

    def _get_close_price(self) -> Decimal:
        """Override: return the close/current market price."""
        raise NotImplementedError

    def _get_open_filled_amount_quote(self) -> Decimal:
        """Override: return the total open filled amount in quote currency."""
        raise NotImplementedError

    def _get_trade_side(self) -> TradeType:
        """Override: return the trade side (BUY or SELL)."""
        raise NotImplementedError

    def _get_cum_fees_from_orders(self) -> Decimal:
        """Override: return the sum of cumulative fees from all orders."""
        raise NotImplementedError
