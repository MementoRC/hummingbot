from typing import Dict

from hummingbot.core.data_type.common import TradeType
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.order_book_message import OrderBookMessage, OrderBookMessageType


class HyperliquidOrderBook(OrderBook):
    @classmethod
    def snapshot_message_from_exchange(
        cls, msg: dict[str, any], timestamp: float, metadata: Dict | None = None
    ) -> OrderBookMessage:
        """
        Creates a snapshot message with the order book snapshot message
        :param msg: the response from the exchange when requesting the order book snapshot
        :param timestamp: the snapshot timestamp
        :param metadata: a dictionary with extra information to add to the snapshot data
        :return: a snapshot message with the snapshot information received from the exchange
        """
        if metadata:
            msg.update(metadata)
        return OrderBookMessage(
            OrderBookMessageType.SNAPSHOT,
            {
                "trading_pair": msg["trading_pair"],
                "update_id": int(msg["time"]),
                "bids": [[float(i["px"]), float(i["sz"])] for i in msg["levels"][0]],
                "asks": [[float(i["px"]), float(i["sz"])] for i in msg["levels"][1]],
            },
            timestamp=timestamp,
        )

    @classmethod
    def diff_message_from_exchange(
        cls, msg: dict[str, any], timestamp: float | None = None, metadata: Dict | None = None
    ) -> OrderBookMessage:
        """
        Creates a diff message with the changes in the order book received from the exchange
        :param msg: the changes in the order book
        :param timestamp: the timestamp of the difference
        :param metadata: a dictionary with extra information to add to the difference data
        :return: a diff message with the changes in the order book notified by the exchange
        """
        if metadata:
            msg.update(metadata)
        return OrderBookMessage(
            OrderBookMessageType.DIFF,
            {
                "trading_pair": msg["trading_pair"],
                "update_id": msg["time"],
                "bids": ([float(i["px"]), float(i["sz"])] for i in msg["levels"][0]),
                "asks": ([float(i["px"]), float(i["sz"])] for i in msg["levels"][1]),
            },
            timestamp=timestamp,
        )

    @classmethod
    def trade_message_from_exchange(cls, msg: dict[str, any], metadata: Dict | None = None):
        """
        Creates a trade message with the information from the trade event sent by the exchange
        :param msg: the trade event details sent by the exchange
        :param metadata: a dictionary with extra information to add to trade message
        :return: a trade message with the details of the trade as provided by the exchange
        """
        if metadata:
            msg.update(metadata)
        return OrderBookMessage(
            OrderBookMessageType.TRADE,
            {
                "trading_pair": msg["trading_pair"],
                "trade_type": float(TradeType.SELL.value) if msg["side"] == "A" else float(TradeType.BUY.value),
                "trade_id": msg["hash"],
                "price": float(msg["px"]),
                "amount": float(msg["sz"]),
            },
            timestamp=msg["time"] * 1e-3,
        )
