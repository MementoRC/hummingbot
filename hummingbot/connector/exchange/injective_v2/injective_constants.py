import sys

from hummingbot.core.api_throttler.data_types import RateLimit
from hummingbot.core.data_type.in_flight_order import OrderState

EXCHANGE_NAME = "injective_v2"

DEFAULT_DOMAIN = ""
TESTNET_DOMAIN = "testnet"

DEFAULT_SUBACCOUNT_INDEX = 0
EXTRA_TRANSACTION_GAS = 20000
DEFAULT_GAS_PRICE = 500000000

EXPECTED_BLOCK_TIME = 1.5
TRANSACTIONS_CHECK_INTERVAL = 3 * EXPECTED_BLOCK_TIME

# Public limit ids
SPOT_MARKETS_LIMIT_ID = "SpotMarkets"
DERIVATIVE_MARKETS_LIMIT_ID = "DerivativeMarkets"
DERIVATIVE_MARKET_LIMIT_ID = "DerivativeMarket"
SPOT_ORDERBOOK_LIMIT_ID = "SpotOrderBookSnapshot"
DERIVATIVE_ORDERBOOK_LIMIT_ID = "DerivativeOrderBookSnapshot"
GET_TRANSACTION_LIMIT_ID = "GetTransaction"
GET_CHAIN_TRANSACTION_LIMIT_ID = "GetChainTransaction"
FUNDING_RATES_LIMIT_ID = "FundingRates"
ORACLE_PRICES_LIMIT_ID = "OraclePrices"
FUNDING_PAYMENTS_LIMIT_ID = "FundingPayments"

# Private limit ids
PORTFOLIO_BALANCES_LIMIT_ID = "AccountPortfolio"
POSITIONS_LIMIT_ID = "Positions"
SPOT_ORDERS_HISTORY_LIMIT_ID = "SpotOrdersHistory"
DERIVATIVE_ORDERS_HISTORY_LIMIT_ID = "DerivativeOrdersHistory"
SPOT_TRADES_LIMIT_ID = "SpotTrades"
DERIVATIVE_TRADES_LIMIT_ID = "DerivativeTrades"
SIMULATE_TRANSACTION_LIMIT_ID = "SimulateTransaction"
SEND_TRANSACTION = "SendTransaction"

NO_LIMIT = sys.maxsize
ONE_SECOND = 1

RATE_LIMITS = [
    RateLimit(limit_id=SPOT_MARKETS_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=DERIVATIVE_MARKETS_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=DERIVATIVE_MARKET_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=SPOT_ORDERBOOK_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=DERIVATIVE_ORDERBOOK_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=GET_TRANSACTION_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=GET_CHAIN_TRANSACTION_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=PORTFOLIO_BALANCES_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=POSITIONS_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=SPOT_ORDERS_HISTORY_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=DERIVATIVE_ORDERS_HISTORY_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=SPOT_TRADES_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=DERIVATIVE_TRADES_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=SIMULATE_TRANSACTION_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=SEND_TRANSACTION, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=FUNDING_RATES_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=ORACLE_PRICES_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
    RateLimit(limit_id=FUNDING_PAYMENTS_LIMIT_ID, limit=NO_LIMIT, time_interval=ONE_SECOND),
]

ORDER_STATE_MAP = {
    "booked": OrderState.OPEN,
    "partial_filled": OrderState.PARTIALLY_FILLED,
    "filled": OrderState.FILLED,
    "canceled": OrderState.CANCELED,
}

ORDER_NOT_FOUND_ERROR_MESSAGE = "order not found"
ACCOUNT_SEQUENCE_MISMATCH_ERROR = "account sequence mismatch"
