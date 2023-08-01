from enum import Enum
from typing import Tuple

from bidict import bidict

from hummingbot.core.api_throttler.data_types import LinkedLimitWeightPair, RateLimit
from hummingbot.core.data_type.in_flight_order import OrderState

EXCHANGE_NAME = "Coinbase Advanced Trade"

CHANGELOG_URL = "https://docs.cloud.coinbase.com/advanced-trade-api/docs/changelog"
LATEST_UPDATE = "2023-JUL-26"
CHANGELOG_HASH = "3a99968df3223eb00502e2059012c9a5"

COINBASE_ADVANCED_TRADE_CLASS_PREFIX = "CoinbaseAdvancedTrade"

DEFAULT_DOMAIN = "com"

HBOT_ORDER_ID_PREFIX = "CBAT-"
MAX_ORDER_ID_LEN = 32
HBOT_BROKER_ID = "Hummingbot"

# Base URL
SIGNIN_URL = "https://api.coinbase.{domain}/v2"
REST_URL = "https://api.coinbase.{domain}/api/v3"
WSS_URL = "wss://advanced-trade-ws.coinbase.{domain}/"

# Coinbase Signin API endpoints
SERVER_TIME_EP = "/time"
EXCHANGE_RATES_USD_EP = "/exchange-rates"
EXCHANGE_RATES_QUOTE_EP = "/exchange-rates?currency={quote_token}"
EXCHANGE_RATES_QUOTE_LIMIT_ID = "ExchangeRatesQuote"
CURRENCIES_EP = "/currencies"
CRYPTO_CURRENCIES_EP = "/currencies/crypto"

SIGNIN_ENDPOINTS = {
    SERVER_TIME_EP,
    EXCHANGE_RATES_USD_EP,
    EXCHANGE_RATES_QUOTE_LIMIT_ID,
    CURRENCIES_EP,
    CRYPTO_CURRENCIES_EP,
}

# Private API endpoints
ALL_PAIRS_EP = "/brokerage/products"
PAIR_TICKER_EP = "/brokerage/products/{product_id}"
PAIR_TICKER_RATE_LIMIT_ID = "PairTicker"
PAIR_TICKER_24HR_EP = "/brokerage/products/{product_id}/ticker"
PAIR_TICKER_24HR_RATE_LIMIT_ID = "PairTicker24Hr"
ORDER_EP = "/brokerage/orders"
BATCH_CANCEL_EP = "/brokerage/orders/batch_cancel"
GET_ORDER_STATUS_EP = "/brokerage/orders/historical/{order_id}"
GET_ORDER_STATUS_RATE_LIMIT_ID = "GetOrderStatus"
GET_STATUS_BATCH_EP = "/brokerage/orders/historical/batch"
FILLS_EP = "/brokerage/orders/historical/fills"
TRANSACTIONS_SUMMARY_EP = "/brokerage/transaction_summary"
ACCOUNTS_LIST_EP = "/brokerage/accounts"
ACCOUNT_EP = "/brokerage/accounts/{account_uuid}"
ACCOUNT_RATE_LIMIT_ID = "Account"
SNAPSHOT_EP = "/brokerage/product_book"

REST_ENDPOINTS = {
    ALL_PAIRS_EP,
    PAIR_TICKER_RATE_LIMIT_ID,
    PAIR_TICKER_24HR_RATE_LIMIT_ID,
    ORDER_EP,
    BATCH_CANCEL_EP,
    GET_ORDER_STATUS_RATE_LIMIT_ID,
    GET_STATUS_BATCH_EP,
    FILLS_EP,
    TRANSACTIONS_SUMMARY_EP,
    ACCOUNTS_LIST_EP,
    ACCOUNT_RATE_LIMIT_ID,
    SNAPSHOT_EP,
}

WS_HEARTBEAT_TIME_INTERVAL = 30


class WebsocketAction(Enum):
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"


# https://docs.cloud.coinbase.com/advanced-trade-api/docs/ws-channels
WS_ORDER_SUBSCRIPTION_KEYS: Tuple[str, ...] = ("level2", "market_trades")
WS_ORDER_SUBSCRIPTION_CHANNELS: bidict[str, str] = bidict({k: k for k in WS_ORDER_SUBSCRIPTION_KEYS})
WS_ORDER_SUBSCRIPTION_CHANNELS["level2"] = "l2_data"

WS_USER_SUBSCRIPTION_KEYS: Tuple[str, ...] = ("user",)
WS_USER_SUBSCRIPTION_CHANNELS: bidict[str, str] = bidict({k: k for k in WS_USER_SUBSCRIPTION_KEYS})

WS_OTHERS_SUBSCRIPTION_KEYS: Tuple[str, ...] = ("ticker", "ticker_batch", "status")
WS_OTHERS_SUBSCRIPTION_CHANNELS: bidict[str, str] = bidict({k: k for k in WS_OTHERS_SUBSCRIPTION_KEYS})

# CoinbaseAdvancedTrade params
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"

# Rate Limit Type
REST_REQUESTS = "REST_REQUESTS"
MAX_REST_REQUESTS_S = 30

SIGNIN_REQUESTS = "SIGNIN_REQUESTS"
MAX_SIGNIN_REQUESTS_H = 10000

WSS_REQUESTS = "WSS_REQUESTS"
MAX_WSS_REQUESTS_S = 750

# Rate Limit time intervals
ONE_SECOND = 1
ONE_MINUTE = 60
ONE_HOUR = 3600
ONE_DAY = 86400

# Order States
ORDER_STATE = {
    "OPEN": OrderState.OPEN,
    "FILLED": OrderState.FILLED,
    "CANCELLED": OrderState.CANCELED,
    "EXPIRED": OrderState.FAILED,
    "FAILED": OrderState.FAILED,
    # Not directly from exchange
    "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
}
# Oddly, order can be in unknown state ???
ORDER_STATUS_NOT_FOUND_ERROR_CODE = "UNKNOWN_ORDER_STATUS"

REST_RATE_LIMITS = [RateLimit(limit_id=endpoint,
                              limit=MAX_REST_REQUESTS_S,
                              time_interval=ONE_SECOND,
                              linked_limits=[LinkedLimitWeightPair(REST_REQUESTS, 1)]) for endpoint in
                    REST_ENDPOINTS]

SIGNIN_RATE_LIMITS = [RateLimit(limit_id=endpoint,
                                limit=MAX_SIGNIN_REQUESTS_H,
                                time_interval=ONE_HOUR,
                                linked_limits=[LinkedLimitWeightPair(REST_REQUESTS, 1)]) for endpoint in
                      SIGNIN_ENDPOINTS]

RATE_LIMITS = [
    # Pools
    RateLimit(limit_id=REST_REQUESTS, limit=MAX_REST_REQUESTS_S, time_interval=ONE_SECOND),
    RateLimit(limit_id=SIGNIN_REQUESTS, limit=MAX_SIGNIN_REQUESTS_H, time_interval=ONE_HOUR),
    RateLimit(limit_id=WSS_REQUESTS, limit=MAX_WSS_REQUESTS_S, time_interval=ONE_SECOND),
] + REST_RATE_LIMITS + SIGNIN_RATE_LIMITS
