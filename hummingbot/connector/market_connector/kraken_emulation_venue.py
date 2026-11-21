import inspect
import logging
from typing import Any, Awaitable, Callable, Optional, Sequence, Union

from data_type_primitives import CancelOutcome, LegSpec, OrderState
from data_type_primitives.common import OrderType
from market_connector.exchanges.kraken.mixins.orders import KRAKEN_ORDER_CAPABILITIES
from strategy_framework.linked_orders import LegFill, LegSnapshot

LinkTag = Union[str, int]

_LOGGER = logging.getLogger(__name__)


def _to_sf_snapshot(snapshot: Any) -> LegSnapshot:
    return LegSnapshot(
        client_order_id=snapshot.client_order_id,
        link_tag=snapshot.link_tag,
        state=snapshot.state,
        filled=snapshot.filled,
    )


class KrakenEmulationVenue:
    """strategy-framework ``EmulationVenue`` over a market-connector Kraken gateway.

    ``target`` is the trading-pair string and is passed to the gateway unchanged.
    """

    def __init__(self, gateway: Any):
        self._gateway = gateway

    def supports_leg(self, target: str, leg: LegSpec) -> bool:
        # Percent trailing-stop-limit needs MC #298 (130918a), which is not in the current pin: deferred.
        if leg.order_type is OrderType.TRAILING_STOP_LIMIT:
            return False
        return leg.order_type in KRAKEN_ORDER_CAPABILITIES.order_types

    async def place_leg(self, target: str, leg: LegSpec, *, link_tag: LinkTag) -> str:
        # LegSpec.price is the limit price for *_LIMIT types and None for trigger-only/market types,
        # which is what the P3 _create_order passes (trigger-only types carry no price there either).
        await self._gateway.place_order(
            target,
            leg.order_type,
            leg.trade_type,
            leg.amount,
            leg.price,
            cl_ord_id=link_tag,
            trigger=leg.trigger,
        )
        # SF matches fills by exact equality on the tag, so return it rather than the txid.
        return str(link_tag)

    async def cancel_leg(self, target: str, client_order_id: str) -> CancelOutcome:
        if await self._gateway.cancel_order(target, cl_ord_id=client_order_id):
            return CancelOutcome.CANCELED
        try:
            snapshot = await self._gateway.get_leg(target, client_order_id=client_order_id)
        except Exception:
            _LOGGER.exception("get_leg failed while disambiguating cancel of %s", client_order_id)
            return CancelOutcome.UNKNOWN
        if snapshot is None or snapshot.state is OrderState.CANCELED:
            return CancelOutcome.NOT_FOUND
        if snapshot.state is OrderState.FILLED:
            return CancelOutcome.ALREADY_FILLED
        return CancelOutcome.UNKNOWN

    async def get_leg(
        self,
        target: str,
        *,
        client_order_id: Optional[str] = None,
        link_tag: Optional[LinkTag] = None,
    ) -> Optional[LegSnapshot]:
        snapshot = await self._gateway.get_leg(target, client_order_id=client_order_id, link_tag=link_tag)
        return None if snapshot is None else _to_sf_snapshot(snapshot)

    async def list_open_legs(self, target: str) -> Sequence[LegSnapshot]:
        return [_to_sf_snapshot(snapshot) for snapshot in await self._gateway.list_open_legs(target)]

    def subscribe(self, on_fill_cb: Callable[[LegFill], Union[Awaitable[None], None]]) -> Callable[[], None]:
        async def _deliver(fill: Any) -> None:
            result = on_fill_cb(
                LegFill(
                    client_order_id=fill.client_order_id,
                    cumulative_filled=fill.cumulative_filled,
                    last_price=fill.last_price,
                    timestamp=fill.timestamp,
                    fill_id=fill.fill_id,
                )
            )
            if inspect.isawaitable(result):
                await result

        return self._gateway.subscribe_leg_fills(_deliver)
