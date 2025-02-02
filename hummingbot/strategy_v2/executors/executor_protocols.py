from decimal import Decimal
from typing import Protocol

from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType


class ExecutorProtocol(Protocol):
    """
    This protocol defines the methods that are required to be implemented by the Executor.

    :param close_type: The type of close that is being executed
    :param _status: The status of the executor.
    """
    close_type: CloseType
    _status: RunnableStatus

    @property
    def status(self) -> RunnableStatus:
        ...

    @property
    def is_closed(self):
        ...

    @property
    def net_pnl_quote(self) -> Decimal:
        ...

    @property
    def net_pnl_pct(self) -> Decimal:
        ...

    @property
    def cum_fees_quote(self) -> Decimal:
        ...

    def stop(self) -> None:
        ...

    def get_in_flight_order(self, connector_name: str, order_id: str) -> InFlightOrder:
        ...

    def place_order(
            self,
            connector_name: str,
            trading_pair: str,
            order_type: OrderType,
            side: TradeType,
            amount: Decimal,
            position_action: PositionAction = PositionAction.NIL,
            price=Decimal("NaN"),
            **kwargs,
    ) -> str:
        ...

    def cancel_order(self, connector_name: str, order_id: str) -> None:
        ...

    def get_trading_rules(self, connector_name: str, trading_pair: str) -> TradingRule:
        ...

    def lock_order_candidate(self, exchange: str, order_candidate: OrderCandidate) -> OrderCandidate:
        ...

    def unlock_order_candidate(self, exchange: str, order_candidate: OrderCandidate) -> OrderCandidate:
        ...
