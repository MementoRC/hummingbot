import asyncio
import logging
import os
from decimal import Decimal
from typing import Dict, List, Set, Tuple

from hummingbot.client.settings import CONF_DIR_PATH, CONF_PREFIX
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

# This implement the mechanics of importing the configuration from a YML file
# It is just convenient to have a separate file that all similar classes can import
from hummingbot.user.user_balances import UserBalances

from .funds_balancer import FundsBalancer
from .markets_yml_config import MarketsYmlConfig
from .trade_route_finder import TradeRouteFinder

lsb_logger = None


class ExampleBalancedLM(ScriptStrategyBase, MarketsYmlConfig):
    """
    Trying to get a better sense of balances and inventory in a common currency (USDT)
    """
    config_filename: str = f"{CONF_DIR_PATH / CONF_PREFIX / os.path.split(__file__)[1].split('.')[0]}.yml"

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self._balancing_trades: Dict = dict()
        self._valid_asset_route: Dict[str, Set] = dict(
            _default_={"BTC", "USDT", "USDC", "KCS", "GT", "DAI", "ADA", "ETH", "AVAX"})
        self._prices: Dict = dict()

        self._async_refresh: float = 45
        self._last_async_refresh_ts: float = 0
        self._tick_time: float = 0

        self._trade_route_finder: Dict = dict()

        # Futures reference called with async on the main event loop (executed after the strategy tick())
        self._prices_fut: Dict = dict()
        self._balances_fut: Dict = dict()
        self._list_futures_done: Dict = dict()
        self._asyncs_called: bool = False

        self._data_ready: bool = False

    @classmethod
    def initialize_from_yml(cls) -> Dict[str, Set[str]]:
        # Load the config or initialize with example configuration
        MarketsYmlConfig.load_from_yml(cls.config_filename)

        # Update the markets with local definition
        if hasattr(cls, 'markets'):
            MarketsYmlConfig.update_markets(cls.markets)

        # Return the markets for initialization of the connectors
        return MarketsYmlConfig.initialize_markets(cls.config_filename)

    def start(self, clock: Clock, timestamp: float):
        self._tick_time = clock.tick_size
        self.logger().info("Starting the strategy")

    def on_tick(self):
        """
        Runs every tick_size seconds, this is the main operation of the strategy.
        - Create proposal (a list of order candidates)
        - Check the account balance and adjust the proposal accordingly (lower order amount if needed)
        - Lastly, execute the proposal on the exchange
        """
        proposals: Dict[str, List[OrderCandidate]] = self._create_proposal()
        if proposals:
            self._execute_proposal(proposals)

    def tick(self, timestamp: float):
        """
        Clock tick entry point, is run every second (on normal tick setting).
        Checks if all connectors are ready, if so the strategy is ready to trade.

        :param timestamp: current tick timestamp
        """
        if not self.ready_to_trade:
            self.ready_to_trade = all(ex.ready for ex in self.connectors.values())

            if not self.ready_to_trade:
                for con in [c for c in self.connectors.values() if not c.ready]:
                    self.logger().warning(f"{con.name} is not ready. Please wait...")
                return
        else:
            if self._last_async_refresh_ts == 0 or self._last_async_refresh_ts < (
                    self.current_timestamp - self._async_refresh):
                self._refresh_balances_prices_routes()
                self._data_ready = True

            if self._data_ready:
                self._last_async_refresh_ts = self.current_timestamp
                self.on_tick()
            else:
                self.logger().warning("Strategy is not ready. Please wait...")
                return

    def did_create_buy_order(self, event: BuyOrderCreatedEvent):
        """
        Method called when the connector notifies a buy order has been created
        """
        self.logger().info(logging.INFO, f"The buy order {event.order_id} has been created")

    def did_create_sell_order(self, event: SellOrderCreatedEvent):
        """
        Method called when the connector notifies a sell order has been created
        """
        self.logger().info(logging.INFO, f"The sell order {event.order_id} has been created")

    def did_fill_order(self, event: OrderFilledEvent):
        """
        Method called when the connector notifies that an order has been partially or totally filled (a trade happened)
        """
        self.logger().info(logging.INFO, f"The order {event.order_id} has been filled")

    def did_fail_order(self, event: MarketOrderFailureEvent):
        """
        Method called when the connector notifies an order has failed
        """
        self.logger().info(logging.INFO, f"The order {event.order_id} failed")

    def did_cancel_order(self, event: OrderCancelledEvent):
        """
        Method called when the connector notifies an order has been cancelled
        """
        self.logger().info(f"The order {event.order_id} has been cancelled")

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        """
        Method called when the connector notifies a buy order has been completed (fully filled)
        """
        self.logger().info(f"The buy order {event.order_id} has been completed")

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        """
        Method called when the connector notifies a sell order has been completed (fully filled)
        """
        self.logger().info(f"The sell order {event.order_id} has been completed")

    def _execute_proposal(self, proposals: Dict[str, List[OrderCandidate]]) -> None:
        """
        Execute the proposals
        """
        for name, list_orders in proposals.items():
            for order_candidate in list_orders:
                if order_candidate.amount > Decimal("0"):
                    if order_candidate.order_side == TradeType.BUY:
                        self.buy(name, order_candidate.trading_pair, order_candidate.amount, order_candidate.order_type,
                                 order_candidate.price)
                    else:
                        self.sell(name, order_candidate.trading_pair, order_candidate.amount,
                                  order_candidate.order_type,
                                  order_candidate.price)

    def _create_proposal(self) -> Dict[str, List[OrderCandidate]]:
        """
        Creates and returns a proposal (a list of order candidate)
        """
        proposal = dict()
        # If the current price (the last close) is below the dip, add a new order candidate to the proposal
        for name, conn in self.connectors.items():
            proposal[name] = list()
            new_pairs = list()
            for trade in self._balancing_trades[name]:
                d = trade['trades']
                for sub_t in [dict(zip(d, v)) for v in zip(*d.values())]:
                    try:
                        price = conn.get_price(sub_t['pairs'], False)
                    except ValueError:
                        self.logger().warning(f"No subscription for {sub_t['pairs']} yet, subscribing")
                        # Add new pair to the list of orderbooks
                        new_pairs.append(sub_t['pairs'])
                        price = sub_t['rates']
                    proposal[name].append(OrderCandidate(sub_t['pairs'],
                                                         False,
                                                         OrderType.LIMIT,
                                                         TradeType.BUY,
                                                         sub_t['amounts'],
                                                         price))
            # Add order books for the new pairs
            self._add_orderbook_pairs(conn, new_pairs)
            self._update_proposal_prices(proposal[name], conn)
            proposal = conn.budget_checker.adjust_candidates(proposal[name], all_or_none=False)

        return proposal

    def _update_proposal_prices(self, proposal: List[OrderCandidate], connector: ConnectorBase) -> None:
        """
        Update the prices of proposal (a list of order candidate)
        """
        for trade in proposal:
            try:
                price = connector.get_price(trade.trading_pair, False)
            except ValueError:
                self.logger().error(f"Unexpected Error: No subscription for {trade.trading_pair} yet")
                raise
            trade.amount = Decimal(price)

    def _add_orderbook_pairs(self, connector: ConnectorBase, pairs: List[str]):
        if self.ready_to_trade and all([p not in connector.trading_pairs for p in pairs]):
            connector.order_book_tracker.add_orderbook_pairs(pairs)

    def _get_connector_price(self, name: str, pair: str) -> Decimal:
        return self.connectors[name].get_price(pair, False)

    def _refresh_balances_prices_routes(self) -> None:
        """
        Calls async methods for all balance & price
        """
        self._data_ready = False
        # We need balances to be updated and prices for both exchange (rather than use the oracle)
        # Submit to the main Event loop  - We get the result after a few ticks and wait till then
        if not self._asyncs_called:
            self._asyncs_called = self._call_asyncs()

        # Wait for the futures to be done
        for name, conn in self.connectors.items():
            if not self._list_futures_done[self._balances_fut[name]] and self._balances_fut[name].done():
                self._list_futures_done[self._balances_fut[name]] = True

            if not self._list_futures_done[self._prices_fut[name]] and self._prices_fut[name].done():
                # Retrieve the prices for the exchange from the completed future
                self._prices[name] = self._prices_fut[name].result()
                # Reset the future for next update
                self._list_futures_done[self._prices_fut[name]] = True

            if self._list_futures_done[self._balances_fut[name]] and self._list_futures_done[self._prices_fut[name]]:
                # Balance and price ready. Compute trades for balancing the portfolio
                self._find_trades_routes(conn, slippage=0.01, fee=0.01)
                # Reset the futures references for the next time we refresh
                del self._list_futures_done[self._prices_fut[name]]
                del self._list_futures_done[self._balances_fut[name]]
                self._balances_fut[name] = None
                self._prices_fut[name] = None

        if all(self._list_futures_done.values()):
            self._asyncs_called = False
            self._data_ready = True

    def _call_asyncs(self) -> bool:
        loop = asyncio.get_event_loop()
        # We need balances to be updated and prices for both exchange (rather than use the oracle)
        # Submit to the main Event loop  - We get the result after a few ticks and wait till then
        for exchange_name, connector in self.connectors.items():
            self._balances_fut[exchange_name] = asyncio.run_coroutine_threadsafe(
                UserBalances.instance().update_exchange_balance(exchange_name), loop)
            self._list_futures_done[self._balances_fut[exchange_name]] = False

            if exchange_name == 'kucoin':
                self._prices_fut['kucoin'] = asyncio.run_coroutine_threadsafe(RateOracle.get_kucoin_prices(), loop)
                self._list_futures_done[self._prices_fut['kucoin']] = False

            elif exchange_name == 'gate_io':
                self._prices_fut['gate_io'] = asyncio.run_coroutine_threadsafe(RateOracle.get_gate_io_prices(), loop)
                self._list_futures_done[self._prices_fut['gate_io']] = False
        return True

    def _find_trades_routes(self, connector: ConnectorBase, slippage: float, fee: float) -> None:
        name = connector.name
        # Compute the balancing proposal
        self._balancing_trades[name] = FundsBalancer.balancing_proposal(self, self._prices, connector)

        # Add the campaign assets as valid routes
        self._valid_asset_route[name] = self._valid_asset_route['_default_'].copy()
        self._valid_asset_route[name].update(self.get_assets_from_config(name))

        # Initialize the route finder
        self._trade_route_finder[name] = TradeRouteFinder(self._prices[name], self._valid_asset_route[name], fee=fee)

        def calculate_order_amount(quantity: Decimal, trade_route: Dict) -> (Tuple[Decimal], Tuple[Decimal]):
            q = [quantity]
            p = [quantity * trade_route['rates'][0]]
            for t_idx, order in enumerate(trade_route['orders']):
                if order == 'sell_quote':
                    # Rewrite as a buy asset
                    q[t_idx] = q[t_idx] * trade_route['rates'][t_idx]
                    p[t_idx] = q[t_idx]

                p[t_idx] *= (Decimal("1") - Decimal(slippage))
                q.append(p[t_idx])
                p.append(0)
            return tuple(q[:-1]), tuple(p[:-1])

        # Find the best route for each trade
        for trade_index, trade in enumerate(self._balancing_trades[name]):
            trade['route'], _, trade['trades'] = \
                self._trade_route_finder[name].best_route(trade['asset'], trade['to'])
            trade['trades']['amounts'], trade['trades']['proceed'] = calculate_order_amount(Decimal(trade['amount']),
                                                                                            trade['trades'])