from decimal import Decimal
from types import SimpleNamespace

import pytest

pytest.importorskip("market_connector")

from data_type_primitives import InFlightOrder, OrderState, TradeType, TradingRules  # noqa: E402
from market_connector.hb_compat.order_events import MARKET_EVENT_IDS  # noqa: E402

from hummingbot.core.data_type.common import OrderType  # noqa: E402

PAIR = "ETH-USDC"
NAN = Decimal("nan")
STATUS_KEYS = (
    "symbols_mapping_initialized",
    "trading_rule_initialized",
    "account_balance",
    "status_polling_initialized",
    "user_stream_initialized",
)


class FakeCore:
    def __init__(self):
        self.status = {key: True for key in STATUS_KEYS}
        self.tracked = []
        self.updated = []
        self.snapshot = None
        self.prices = {}
        self.trading_rules = {
            PAIR: TradingRules(
                trading_pair=PAIR,
                min_order_size=Decimal("0.01"),
                min_price_increment=Decimal("0.01"),
                min_base_amount_increment=Decimal("0.001"),
            )
        }
        self.fail_refresh = False
        self.bus = None
        self.in_flight_orders = {}
        self.finished = {}
        self.txids = {}
        self.order_calls = []
        self.cancel_calls = []
        self._counter = 0

    @property
    def status_dict(self):
        return dict(self.status)

    @property
    def ready(self):
        return all(self.status.values())

    def track_pairs(self, pairs):
        self.tracked.extend(pairs)

    async def update_order_book(self, pair):
        self.updated.append(pair)
        if self.fail_refresh:
            raise RuntimeError("boom")

    def get_order_book(self, pair):
        return self.snapshot

    def get_price_by_type(self, pair, price_type):
        return self.prices.get(price_type, NAN)

    def get_balance(self, asset):
        return {"ETH": Decimal("2")}.get(asset, Decimal(0))

    def get_available_balance(self, asset):
        return {"ETH": Decimal("1.5")}.get(asset, Decimal(0))

    def quantize_order_amount(self, pair, amount):
        return amount.quantize(Decimal("0.001"))

    def quantize_order_price(self, pair, price):
        return price.quantize(Decimal("0.01"))

    def get_order_price_quantum(self, pair, price):
        return Decimal("0.01")

    def get_order_size_quantum(self, pair, amount):
        return Decimal("0.001")

    # --- order API -------------------------------------------------------------------------------------------

    def _place(self, side, pair, amount, order_type, price, **extras):
        self._counter += 1
        cid = f"hbot-{side}-{self._counter}"
        self.order_calls.append((side, pair, amount, order_type, price, extras))
        return cid

    def buy(self, trading_pair, amount, order_type=OrderType.LIMIT, price=NAN, **extras):
        return self._place("buy", trading_pair, amount, order_type, price, **extras)

    def sell(self, trading_pair, amount, order_type=OrderType.LIMIT, price=NAN, **extras):
        return self._place("sell", trading_pair, amount, order_type, price, **extras)

    def cancel(self, trading_pair, client_order_id):
        self.cancel_calls.append((trading_pair, client_order_id))
        return client_order_id

    def fetch_order(self, client_order_id):
        return self.in_flight_orders.get(client_order_id) or self.finished.get(client_order_id)

    def client_order_id_for_txid(self, txid):
        return self.txids.get(txid)

    def add_order(self, order, *, finished=False):
        (self.finished if finished else self.in_flight_orders)[order.client_order_id] = order
        if order.exchange_order_id:
            self.txids[order.exchange_order_id] = order.client_order_id

    def emit(self, event):
        self.bus.publish(MARKET_EVENT_IDS[type(event).__name__], event)


class FakeGateway:
    """Records calls; ``*_return`` / ``*_raises`` configure the outcome of each method."""

    def __init__(self):
        self.calls = []
        self.place_return = "TXID-1"
        self.place_raises = None
        self.cancel_return = True
        self.cancel_raises = None
        self.get_leg_return = None
        self.get_leg_raises = None
        self.open_legs = []
        self.callbacks = []
        self.unsubscribed = 0

    async def place_order(
        self,
        trading_pair,
        order_type,
        side,
        amount,
        price=None,
        *,
        limit_price=None,
        price_in_percent=False,
        userref=None,
        cl_ord_id=None,
        trigger=None,
    ):
        kwargs = {
            "limit_price": limit_price,
            "price_in_percent": price_in_percent,
            "userref": userref,
            "cl_ord_id": cl_ord_id,
            "trigger": trigger,
        }
        self.calls.append(("place_order", trading_pair, order_type, side, amount, price, kwargs))
        if self.place_raises is not None:
            raise self.place_raises
        return self.place_return

    async def cancel_order(self, trading_pair, txid=None, *, cl_ord_id=None):
        self.calls.append(("cancel_order", trading_pair, txid, cl_ord_id))
        if self.cancel_raises is not None:
            raise self.cancel_raises
        return self.cancel_return

    async def get_leg(self, target, *, client_order_id=None, link_tag=None):
        self.calls.append(("get_leg", target, client_order_id, link_tag))
        if self.get_leg_raises is not None:
            raise self.get_leg_raises
        return self.get_leg_return

    async def list_open_legs(self, target, *, group=None):
        self.calls.append(("list_open_legs", target))
        return list(self.open_legs)

    def subscribe_leg_fills(self, callback):
        self.calls.append(("subscribe_leg_fills",))
        self.callbacks.append(callback)

        def unsubscribe():
            self.unsubscribed += 1

        return unsubscribe

    async def emit_fill(self, fill):
        for callback in list(self.callbacks):
            await callback(fill)


class FakeHandle:
    def __init__(self, core, gateway=None):
        self.core = core
        self.gateway = gateway if gateway is not None else FakeGateway()
        self.started = 0
        self.closed = 0

    async def start(self):
        self.started += 1

    async def close(self):
        self.closed += 1


class FakeDescriptor:
    default_maker_fee = Decimal("0.0025")
    default_taker_fee = Decimal("0.004")
    order_capabilities = SimpleNamespace(
        order_types=frozenset(
            {
                OrderType.MARKET,
                OrderType.LIMIT,
                OrderType.STOP_LOSS,
                OrderType.TAKE_PROFIT,
                OrderType.STOP_LOSS_LIMIT,
                OrderType.TAKE_PROFIT_LIMIT,
                OrderType.TRAILING_STOP,
                OrderType.TRAILING_STOP_LIMIT,
            }
        )
    )

    def __init__(self):
        self.core = FakeCore()
        self.handle = FakeHandle(self.core)
        self.create_calls = []

    async def create(self, config, *, event_bus, **core_options):
        self.create_calls.append((dict(config), event_bus, core_options))
        self.core.bus = event_bus
        return self.handle


def make_order(cid="c1", state=OrderState.OPEN, side=None, amount="1", price="100", txid="TX1"):
    return InFlightOrder(
        client_order_id=cid,
        trading_pair=PAIR,
        order_type=OrderType.LIMIT,
        trade_type=side or TradeType.BUY,
        amount=Decimal(amount),
        creation_timestamp=1700000000.0,
        price=Decimal(price),
        exchange_order_id=txid,
        initial_state=state,
    )
