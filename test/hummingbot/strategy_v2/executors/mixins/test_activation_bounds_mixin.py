"""Tests for ActivationBoundsMixin."""

from decimal import Decimal
from unittest.mock import MagicMock

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.strategy_v2.executors.mixins.activation_bounds import ActivationBoundsMixin


class MockExecutorWithActivationBounds(ActivationBoundsMixin):
    """Mock executor with activation bounds mixin."""

    def __init__(self, activation_bounds=None, mid_price=Decimal("100")):
        self.config = MagicMock()
        self.config.activation_bounds = activation_bounds
        self.config.connector_name = "binance"
        self.config.trading_pair = "BTC-USDT"
        self._mid_price = mid_price

    def get_price(self, connector_name, trading_pair, price_type=None):
        return self._mid_price


class TestActivationBoundsMixin:
    # --- No bounds configured ---

    def test_no_bounds_returns_true(self):
        executor = MockExecutorWithActivationBounds(activation_bounds=None)
        assert executor._is_within_activation_bounds(Decimal("100"), TradeType.BUY, OrderType.LIMIT) is True

    # --- Limit-type orders ---

    def test_limit_buy_within_bounds(self):
        """BUY limit: order_price >= mid_price * (1 - bounds[0])"""
        executor = MockExecutorWithActivationBounds(
            activation_bounds=[Decimal("0.05"), Decimal("0.10")],
            mid_price=Decimal("100"),
        )
        # order_price=96 >= 100*(1-0.05)=95 → True
        assert executor._is_within_activation_bounds(Decimal("96"), TradeType.BUY, OrderType.LIMIT) is True

    def test_limit_buy_outside_bounds(self):
        executor = MockExecutorWithActivationBounds(
            activation_bounds=[Decimal("0.05"), Decimal("0.10")],
            mid_price=Decimal("100"),
        )
        # order_price=94 >= 100*(1-0.05)=95 → False
        assert executor._is_within_activation_bounds(Decimal("94"), TradeType.BUY, OrderType.LIMIT) is False

    def test_limit_sell_within_bounds(self):
        """SELL limit: order_price <= mid_price * (1 + bounds[0])"""
        executor = MockExecutorWithActivationBounds(
            activation_bounds=[Decimal("0.05"), Decimal("0.10")],
            mid_price=Decimal("100"),
        )
        # order_price=104 <= 100*(1+0.05)=105 → True
        assert executor._is_within_activation_bounds(Decimal("104"), TradeType.SELL, OrderType.LIMIT) is True

    def test_limit_sell_outside_bounds(self):
        executor = MockExecutorWithActivationBounds(
            activation_bounds=[Decimal("0.05"), Decimal("0.10")],
            mid_price=Decimal("100"),
        )
        # order_price=106 <= 100*(1+0.05)=105 → False
        assert executor._is_within_activation_bounds(Decimal("106"), TradeType.SELL, OrderType.LIMIT) is False

    def test_limit_maker_within_bounds(self):
        """LIMIT_MAKER is a limit type too."""
        executor = MockExecutorWithActivationBounds(
            activation_bounds=[Decimal("0.05"), Decimal("0.10")],
            mid_price=Decimal("100"),
        )
        assert executor._is_within_activation_bounds(Decimal("96"), TradeType.BUY, OrderType.LIMIT_MAKER) is True

    # --- Market-type orders ---

    def test_market_buy_within_bounds(self):
        """BUY market: order_price*(1-bounds[0]) <= mid <= order_price*(1+bounds[1])"""
        executor = MockExecutorWithActivationBounds(
            activation_bounds=[Decimal("0.05"), Decimal("0.10")],
            mid_price=Decimal("100"),
        )
        # order_price=100: 100*(1-0.05)=95 <= 100 <= 100*(1+0.10)=110 → True
        assert executor._is_within_activation_bounds(Decimal("100"), TradeType.BUY, OrderType.MARKET) is True

    def test_market_buy_below_lower_bound(self):
        executor = MockExecutorWithActivationBounds(
            activation_bounds=[Decimal("0.05"), Decimal("0.10")],
            mid_price=Decimal("80"),
        )
        # order_price=100: 95 <= 80 → False
        assert executor._is_within_activation_bounds(Decimal("100"), TradeType.BUY, OrderType.MARKET) is False

    def test_market_buy_above_upper_bound(self):
        executor = MockExecutorWithActivationBounds(
            activation_bounds=[Decimal("0.05"), Decimal("0.10")],
            mid_price=Decimal("120"),
        )
        # order_price=100: 80 <= 120 <= 110 → False
        assert executor._is_within_activation_bounds(Decimal("100"), TradeType.BUY, OrderType.MARKET) is False

    def test_market_sell_within_bounds(self):
        """SELL market: order_price*(1-bounds[1]) <= mid <= order_price*(1+bounds[0])"""
        executor = MockExecutorWithActivationBounds(
            activation_bounds=[Decimal("0.05"), Decimal("0.10")],
            mid_price=Decimal("100"),
        )
        # order_price=100: 100*(1-0.10)=90 <= 100 <= 100*(1+0.05)=105 → True
        assert executor._is_within_activation_bounds(Decimal("100"), TradeType.SELL, OrderType.MARKET) is True

    def test_market_sell_outside_bounds(self):
        executor = MockExecutorWithActivationBounds(
            activation_bounds=[Decimal("0.05"), Decimal("0.10")],
            mid_price=Decimal("120"),
        )
        # order_price=100: 90 <= 120 <= 105 → False
        assert executor._is_within_activation_bounds(Decimal("100"), TradeType.SELL, OrderType.MARKET) is False

    # --- Boundary edge cases ---

    def test_limit_buy_exactly_at_boundary(self):
        """Exact boundary value should be included (>=)."""
        executor = MockExecutorWithActivationBounds(
            activation_bounds=[Decimal("0.05"), Decimal("0.10")],
            mid_price=Decimal("100"),
        )
        # order_price=95 >= 100*(1-0.05)=95 → True (equal)
        assert executor._is_within_activation_bounds(Decimal("95"), TradeType.BUY, OrderType.LIMIT) is True

    def test_market_buy_exactly_at_lower_boundary(self):
        """Exact boundary for market type should be included (<=)."""
        executor = MockExecutorWithActivationBounds(
            activation_bounds=[Decimal("0.05"), Decimal("0.10")],
            mid_price=Decimal("95"),
        )
        # order_price=100: 100*(1-0.05)=95 <= 95 → True (equal)
        assert executor._is_within_activation_bounds(Decimal("100"), TradeType.BUY, OrderType.MARKET) is True
