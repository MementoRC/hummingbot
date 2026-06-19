from decimal import Decimal
import os

from pydantic import Field
from remote_iface import ExternalEventFactory, ExternalTopicFactory

from hummingbot.core.data_type.common import MarketDict, OrderType
from hummingbot.core.event.events import BuyOrderCreatedEvent, MarketOrderFailureEvent, SellOrderCreatedEvent
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase


class ExternalEventsExampleConfig(StrategyV2ConfigBase):
    script_file_name: str = os.path.basename(__file__)
    exchange: str = Field(default="kucoin_paper_trade")
    trading_pair: str = Field(default="BTC-USDT")

    def update_markets(self, markets: MarketDict) -> MarketDict:
        markets[self.exchange] = markets.get(self.exchange, set()) | {self.trading_pair}
        return markets


class ExternalEventsExample(StrategyV2Base):
    """
    Simple script that uses the external events plugin to create buy and sell
    market orders.
    """

    # ---- Using callback functions ----
    # ----------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from hummingbot.client.hummingbot_application import HummingbotApplication

        app = HummingbotApplication.main_application()
        if app is None or app._mqtt is None:
            self.eevents = None
            self.etopic_queue = None
            self.listener = None
            return
        # hbot/{id}/external/events/*
        self.eevents = ExternalEventFactory.create_queue(app._mqtt, "*")
        # hbot/{id}/test/a
        self.etopic_queue = ExternalTopicFactory.create_queue(app._mqtt, "test/a")
        ExternalEventFactory.create_async(app._mqtt, "*", self.on_event)
        self.listener = ExternalTopicFactory.create_async(app._mqtt, "test/a", self.on_message)

    def on_event(self, msg, name):
        self.logger().info(f"OnEvent Callback fired: {name} -> {msg}")

    def on_message(self, msg, topic):
        self.logger().info(f"Topic Message Callback fired: {topic} -> {msg}")

    async def on_stop(self):
        from hummingbot.client.hummingbot_application import HummingbotApplication

        app = HummingbotApplication.main_application()
        if app is None or app._mqtt is None:
            return
        ExternalEventFactory.remove_listener(app._mqtt, "*", self.on_event)
        if self.listener is not None:
            ExternalTopicFactory.remove_listener(self.listener)

    # ----------------------------------

    def on_tick(self):
        if self.eevents is not None:
            while len(self.eevents) > 0:
                event = self.eevents.popleft()
                self.logger().info(f"External Event in Queue: {event}")
                # event = (name, msg)
                if event[0] == "order.market":
                    if event[1].data["type"] in ("buy", "Buy", "BUY"):
                        self.execute_order(Decimal(event[1].data["amount"]), True)
                    elif event[1].data["type"] in ("sell", "Sell", "SELL"):
                        self.execute_order(Decimal(event[1].data["amount"]), False)
        if self.etopic_queue is not None:
            while len(self.etopic_queue) > 0:
                entry = self.etopic_queue.popleft()
                self.logger().info(f"Topic Message in Queue: {entry[0]} -> {entry[1]}")

    def execute_order(self, amount: Decimal, is_buy: bool):
        if is_buy:
            self.buy(self.config.exchange, self.config.trading_pair, amount, OrderType.MARKET)
        else:
            self.sell(self.config.exchange, self.config.trading_pair, amount, OrderType.MARKET)

    def did_create_buy_order(self, event: BuyOrderCreatedEvent):
        """
        Method called when the connector notifies a buy order has been created
        """
        self.logger().info(f"The buy order {event.order_id} has been created")

    def did_create_sell_order(self, event: SellOrderCreatedEvent):
        """
        Method called when the connector notifies a sell order has been created
        """
        self.logger().info(f"The sell order {event.order_id} has been created")

    def did_fail_order(self, event: MarketOrderFailureEvent):
        """
        Method called when the connector notifies an order has failed
        """
        self.logger().info(f"The order {event.order_id} failed")
