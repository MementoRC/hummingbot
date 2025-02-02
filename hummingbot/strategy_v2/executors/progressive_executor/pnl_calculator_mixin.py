import math
from decimal import Decimal

from hummingbot.core.data_type.common import TradeType

from .protocols import ProgressiveOrderPNLProtocol, ProgressiveOrderProtocol

_60 = Decimal("60")
_24 = Decimal("24")
_365 = Decimal("365")

class PNLCalculatorMixin:
    """
    PNLCalculatorMixin provides methods and properties for calculating profit and loss (PNL) metrics related to trading activities.

    This mixin includes calculations for trade PNL percentages, trade PNL in quote assets, net PNL, cumulative fees, and target PNL yield. Implementing classes should use these methods to effectively manage and report on trading performance.

    Properties:
        trade_pnl_pct (Decimal): The trade PNL percentage, calculated as the sum of realized and unrealized PNL relative to the entry price.
        trade_pnl_quote (Decimal): The trade PNL in quote asset, derived from the trade PNL percentage.

    Methods:
        get_net_pnl_quote() -> Decimal: Calculates the net PNL in quote asset by subtracting cumulative fees from trade PNL.
        get_cum_fees_quote() -> Decimal: Calculates the cumulative fees in quote asset from open, close, and realized orders.
        get_net_pnl_pct() -> Decimal: Calculates the net PNL percentage based on the net PNL in quote asset.
        get_target_pnl_yield() -> Decimal: Returns the target PNL yield based on the APR yield and time elapsed since the last timestamp.
    """

    @staticmethod
    def _realized_pnl(pop: ProgressiveOrderProtocol, side: int) -> Decimal:
        """
        Calculate the trade pnl (Pure pnl without fees)

        :return: The trade pnl percentage.
        """
        return sum(
            (o.executed_amount_base * side * (o.average_executed_price - pop.entry_price)
             for o in pop.realized_orders),
            start=Decimal("0")
        )

    @staticmethod
    def _unrealized_pnl(pop: ProgressiveOrderProtocol, side: int) -> Decimal:
        """
        Calculate the trade pnl (Pure pnl without fees)

        :return: The trade pnl percentage.
        """
        return pop.unrealized_filled_amount * side * (pop.close_price - pop.entry_price)

    @property
    def trade_pnl_pct(self: ProgressiveOrderProtocol) -> Decimal:
        """
        Calculate the trade pnl (Pure pnl without fees)

        :return: The trade pnl percentage.
        """
        if self.open_filled_amount == Decimal("0"):
            return Decimal("0")
        side: int = 1 if self.side == TradeType.BUY else -1
        realized: Decimal = PNLCalculatorMixin._realized_pnl(self, side)
        unrealized: Decimal = PNLCalculatorMixin._unrealized_pnl(self, side)
        return (realized + unrealized) / (self.entry_price * self.open_filled_amount)
        # cum_fees: Decimal = self.cum_fees_quote
        # if self.open_order and self.close_order and self.open_order.is_done and not self.close_order.is_done:
        #     cum_fees += self.open_order.cum_fees_quote
        # return (realized + unrealized - cum_fees) / (self.entry_price * self.open_filled_amount)

    @property
    def trade_pnl_quote(self: ProgressiveOrderPNLProtocol) -> Decimal:
        """
        Calculate the trade pnl in quote asset

        :return: The trade pnl in quote asset.
        """
        return self.trade_pnl_pct * self.open_filled_amount * self.entry_price

    def get_net_pnl_quote(self: ProgressiveOrderPNLProtocol) -> Decimal:
        """
        Calculate the net pnl in quote asset

        :return: The net pnl in quote asset.
        """
        return self.trade_pnl_quote - self.cum_fees_quote

    def get_cum_fees_quote(self: ProgressiveOrderPNLProtocol) -> Decimal:
        """
        Calculate the cumulative fees in quote asset

        :return: The cumulative fees in quote asset.
        """
        orders = [self.open_order, self.close_order, *self.realized_orders]
        return sum((order.cum_fees_quote for order in orders if order), start=Decimal("0"))

    def get_net_pnl_pct(self: ProgressiveOrderPNLProtocol) -> Decimal:
        """
        Calculate the net pnl percentage

        :return: The net pnl percentage.
        """
        return self.net_pnl_quote / self.open_filled_amount_quote if self.open_filled_amount_quote != Decimal(
            "0") else Decimal("0")

    def get_target_pnl_yield(self: ProgressiveOrderPNLProtocol) -> Decimal:
        """
        Returns the target pnl yield

        :return: Target pnl yield
        """
        time_lapsed: Decimal = Decimal(self.current_timestamp - self.config.timestamp)
        pnl: Decimal = self.config.triple_barrier_config.apr_yield * time_lapsed / (_60 * _60 * _24 * _365)
        return Decimal("0") if math.isnan(pnl) else pnl
