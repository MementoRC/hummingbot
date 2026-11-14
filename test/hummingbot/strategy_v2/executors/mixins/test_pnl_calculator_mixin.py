"""Tests for PNLCalculatorMixin."""

from decimal import Decimal

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.mixins.pnl_calculator import PNLCalculatorMixin


class MockPNLExecutor(PNLCalculatorMixin):
    """Mock executor implementing PNL calculator template methods."""

    def __init__(
        self,
        entry_price=Decimal("100"),
        close_price=Decimal("110"),
        filled_quote=Decimal("1000"),
        side=TradeType.BUY,
        cum_fees=Decimal("2"),
    ):
        self._entry = entry_price
        self._close = close_price
        self._filled_quote = filled_quote
        self._side = side
        self._cum_fees = cum_fees

    # Simulate ExecutorBase properties that PNLCalculatorMixin.get_net_pnl_quote
    # accesses via self.cum_fees_quote and self.net_pnl_quote
    @property
    def cum_fees_quote(self):
        return self.get_cum_fees_quote()

    @property
    def net_pnl_quote(self):
        return self.get_net_pnl_quote()

    def _get_entry_price(self):
        return self._entry

    def _get_close_price(self):
        return self._close

    def _get_open_filled_amount_quote(self):
        return self._filled_quote

    def _get_trade_side(self):
        return self._side

    def _get_cum_fees_from_orders(self):
        return self._cum_fees


class TestPNLCalculatorMixin:
    def test_buy_positive_pnl(self):
        executor = MockPNLExecutor(
            entry_price=Decimal("100"),
            close_price=Decimal("110"),
            side=TradeType.BUY,
        )
        assert executor.trade_pnl_pct == Decimal("0.1")  # (110-100)/100

    def test_buy_negative_pnl(self):
        executor = MockPNLExecutor(
            entry_price=Decimal("100"),
            close_price=Decimal("90"),
            side=TradeType.BUY,
        )
        assert executor.trade_pnl_pct == Decimal("-0.1")  # (90-100)/100

    def test_sell_positive_pnl(self):
        executor = MockPNLExecutor(
            entry_price=Decimal("100"),
            close_price=Decimal("90"),
            side=TradeType.SELL,
        )
        assert executor.trade_pnl_pct == Decimal("0.1")  # (100-90)/100

    def test_sell_negative_pnl(self):
        executor = MockPNLExecutor(
            entry_price=Decimal("100"),
            close_price=Decimal("110"),
            side=TradeType.SELL,
        )
        assert executor.trade_pnl_pct == Decimal("-0.1")  # (100-110)/100

    def test_zero_entry_price_returns_zero(self):
        executor = MockPNLExecutor(entry_price=Decimal("0"))
        assert executor.trade_pnl_pct == Decimal("0")

    def test_trade_pnl_quote(self):
        executor = MockPNLExecutor(
            entry_price=Decimal("100"),
            close_price=Decimal("110"),
            filled_quote=Decimal("1000"),
            side=TradeType.BUY,
        )
        # pnl_pct = 0.1, * 1000 = 100
        assert executor.trade_pnl_quote == Decimal("100")

    def test_cum_fees_quote(self):
        executor = MockPNLExecutor(cum_fees=Decimal("5.5"))
        assert executor.get_cum_fees_quote() == Decimal("5.5")

    def test_net_pnl_quote(self):
        executor = MockPNLExecutor(
            entry_price=Decimal("100"),
            close_price=Decimal("110"),
            filled_quote=Decimal("1000"),
            side=TradeType.BUY,
            cum_fees=Decimal("2"),
        )
        # trade_pnl_quote=100, cum_fees=2, net=98
        assert executor.get_net_pnl_quote() == Decimal("98")

    def test_net_pnl_pct(self):
        executor = MockPNLExecutor(
            entry_price=Decimal("100"),
            close_price=Decimal("110"),
            filled_quote=Decimal("1000"),
            side=TradeType.BUY,
            cum_fees=Decimal("2"),
        )
        # net_pnl_quote=98, filled_quote=1000, net_pnl_pct=0.098
        assert executor.get_net_pnl_pct() == Decimal("0.098")

    def test_net_pnl_pct_zero_filled(self):
        executor = MockPNLExecutor(filled_quote=Decimal("0"))
        assert executor.get_net_pnl_pct() == Decimal("0")

    def test_flat_price_zero_pnl(self):
        executor = MockPNLExecutor(
            entry_price=Decimal("100"),
            close_price=Decimal("100"),
            side=TradeType.BUY,
        )
        assert executor.trade_pnl_pct == Decimal("0")
