from abc import ABC
from datetime import datetime
from typing import List, Optional

from pydantic import Field
from pydantic.class_validators import validator

from hummingbot.core.utils.class_registry import ClassRegistry

from ..cat_utilities.cat_pydantic_for_json import PydanticMockableForJson
from .cat_api_request_bases import _RequestGET, _RequestPOST, _RequestProtocolAbstract
from .cat_api_v3_enums import (
    CoinbaseAdvancedTradeExchangeOrderStatusEnum,
    CoinbaseAdvancedTradeExchangeOrderTypeEnum,
    CoinbaseAdvancedTradeOrderSide,
    CoinbaseAdvancedTradeRateLimitType as _RateLimitType,
)
from .cat_api_v3_order_types import CoinbaseAdvancedTradeAPIOrderConfiguration
from .cat_data_types_utilities import UnixTimestampSecondFieldToDatetime, UnixTimestampSecondFieldToStr
from .cat_endpoint_rate_limit import CoinbaseAdvancedTradeEndpointRateLimit


class CoinbaseAdvancedTradeRequestException(Exception):
    pass


class CoinbaseAdvancedTradeRequest(
    ClassRegistry,
    _RequestProtocolAbstract,
    ABC,  # Defines the method that the subclasses must implement to match the Protocol
):
    BASE_ENDPOINT: str = "api/v3/brokerage"  # "/" is added between the base URI and the endpoint

    # This definition allows CoinbaseAdvancedTradeRequest to be used as a Protocol that
    # receives arguments in the constructor. The main purpose of this class is to be
    # used as a Base class for similar subclasses
    def __init__(self, *args, **kwargs):
        if super().__class__ != object:
            super().__init__(**kwargs)

    @classmethod
    def short_class_name(cls) -> str:
        # This method helps clarify that a subclass of this ClassRegistry will
        # have a method called `short_class_name` that returns a string of the
        # class name without the base class (CoinbaseAdvancedTradeRequest) name.
        raise CoinbaseAdvancedTradeRequestException(
            "The method short_class_name should have been dynamically created by ClassRegistry.\n"
            "This exception indicates that the class hierarchy is not correctly implemented and"
            "the CoinbaseAdvancedTradeRequest.short_class_name() was called instead.\n"
        )

    @classmethod
    def linked_limit(cls) -> _RateLimitType:
        return _RateLimitType.REST  # This is either REST, WSS or SIGNIN, as Rate Limit categories


class CoinbaseAdvancedTradeListAccountsRequest(
    _RequestGET,  # GET method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for ListAccountsEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getaccounts
    ```json
    {
        "limit": 0,
        "cursor": "string"
    }
    ```
    """
    # TODO: Verify that the limit is 49 by default and 250 max.
    limit: Optional[int] = Field(None, lt=251, description='A pagination limit with default of 49 and maximum of 250. '
                                                           'If has_next is true, additional orders are available to '
                                                           'be fetched with pagination and the cursor value in the '
                                                           'response can be passed as cursor parameter in the '
                                                           'subsequent request.')
    cursor: Optional[str] = Field(None, description='Cursor used for pagination. When provided, the response returns '
                                                    'responses after this cursor.')

    def endpoint(self) -> str:
        return "accounts"


class CoinbaseAdvancedTradeGetAccountRequest(
    _RequestGET,  # GET method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for GetAccountEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getaccount
    ```json
    {
        "account_uuid": "string"
    }
    ```
    """
    account_uuid: str = Field(..., extra={'path_param': True}, description="The account's UUID.")

    def endpoint(self) -> str:
        return f"accounts/{self.account_uuid}"


class CoinbaseAdvancedTradeCreateOrderRequest(
    _RequestPOST,  # POST method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for CreateOrderEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_postorder
    ```json
    {
        "client_order_id": "string",
        "product_id": "string",
        "side": "UNKNOWN_ORDER_SIDE",
        "order_configuration": {
            "market_market_ioc": {
            "quote_size": "10.00",
            "base_size": "0.001"
          },
          "limit_limit_gtc": {
            "base_size": "0.001",
            "limit_price": "10000.00",
            "post_only": false
          },
          "limit_limit_gtd": {
            "base_size": "0.001",
            "limit_price": "10000.00",
            "end_time": "2021-05-31T09:59:59Z",
            "post_only": false
          },
          "stop_limit_stop_limit_gtc": {
            "base_size": "0.001",
            "limit_price": "10000.00",
            "stop_price": "20000.00",
            "stop_direction": "UNKNOWN_STOP_DIRECTION"
          },
          "stop_limit_stop_limit_gtd": {
            "base_size": "0.001",
            "limit_price": "10000.00",
            "stop_price": "20000.00",
            "end_time": "2021-05-31T09:59:59Z",
            "stop_direction": "UNKNOWN_STOP_DIRECTION"
          }
        }
    }
    ```
    """
    client_order_id: str = Field(..., description='Client set unique uuid for this order')
    product_id: str = Field(..., description="The product this order was created for e.g. 'BTC-USD'")
    side: CoinbaseAdvancedTradeOrderSide = Field(None, description='Possible values: [UNKNOWN_ORDER_SIDE, BUY, SELL]')
    order_configuration: CoinbaseAdvancedTradeAPIOrderConfiguration = Field(None, description='Order configuration')

    def endpoint(self) -> str:
        return "orders"


class CoinbaseAdvancedTradeCancelOrdersRequest(
    _RequestPOST,  # POST method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for CancelOrdersEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_cancelorders
    ```json
    {
        "order_ids": [
            "string"
        ]
    }
    ```
    """
    order_ids: List[str] = Field(..., description='The IDs of orders cancel requests should be initiated for')

    def endpoint(self) -> str:
        return "orders/batch_cancel"


class CoinbaseAdvancedTradeListOrdersRequest(
    _RequestGET,  # GET method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for ListOrdersEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_gethistoricalorders
    ```json
    {
        "product_id": "string",
        "order_status": [
            "OPEN"
        ],
        "limit": 0,
        "start_date": "2021-07-01T00:00:00.000Z",
        "end_date": "2021-07-01T00:00:00.000Z",
        "user_native_currency": "string",
        "order_type": "UNKNOWN_ORDER_TYPE",
        "order_side": "UNKNOWN_ORDER_SIDE",
        "cursor": "string",
        "product_type": "string",
        "order_placement_source": "string"
    }
    ```
    """
    product_id: Optional[str] = Field(None, description='Optional string of the product ID. Defaults to null, '
                                                        'or fetch for all products.')
    order_status: Optional[List[CoinbaseAdvancedTradeExchangeOrderStatusEnum]] = Field(None,
                                                                                       description='A list of order '
                                                                                                   'statuses.')
    limit: Optional[int] = Field(None, description='A pagination limit with no default set. If has_next is true, '
                                                   'additional orders are available to be fetched with pagination; '
                                                   'also the cursor value in the response can be passed as cursor '
                                                   'parameter in the subsequent request.')
    start_date: Optional[UnixTimestampSecondFieldToDatetime] = Field(None,
                                                                     description='Start date to fetch orders from, '
                                                                                 'inclusive.')
    end_date: Optional[UnixTimestampSecondFieldToDatetime] = Field(None,
                                                                   description='An optional end date for the query '
                                                                               'window, exclusive. If'
                                                                               'provided only orders with creation '
                                                                               'time before this date'
                                                                               'will be returned.')
    user_native_currency: Optional[str] = Field(None,
                                                description='String of the users native currency. Default is USD.')
    order_type: Optional[CoinbaseAdvancedTradeExchangeOrderTypeEnum] = Field(None, description='Type of orders to '
                                                                                               'return. Default is to'
                                                                                               ' return all order '
                                                                                               'types.')
    order_side: Optional[CoinbaseAdvancedTradeOrderSide] = Field(None, description='Only orders matching this side '
                                                                                   'are returned. Default is to '
                                                                                   'return all sides.')
    cursor: Optional[str] = Field(None, description='Cursor used for pagination. When provided, the response returns '
                                                    'responses after this cursor.')
    product_type: Optional[str] = Field(None, description='Only orders matching this product type are returned. '
                                                          'Default is to return all product types.')
    order_placement_source: Optional[str] = Field(None, description='Only orders matching this placement source are '
                                                                    'returned. Default is to return RETAIL_ADVANCED '
                                                                    'placement source.')

    def endpoint(self) -> str:
        return "orders/historical/batch"

    @validator('order_status', pre=True)
    def validate_order_status(cls, v):
        if CoinbaseAdvancedTradeExchangeOrderStatusEnum.OPEN in v and len(v) > 1:
            raise ValueError('OPEN is not allowed with other order statuses')
        return v

    class Config:
        json_encoders = {
            # TODO: Check on Coinbase Help for correct format
            datetime: lambda v: v.strftime("%Y-%m-%dT%H:%M:%S") + f".{v.microsecond // 1000:03d}Z",
        }


class CoinbaseAdvancedTradeGetOrderRequest(
    _RequestGET,  # GET method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for GetOrderEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_gethistoricalorder
    ```json
    {
        "order_id": "string",
        "client_order_id": "string",
        "user_native_currency": "string"
    }
    ```
    """
    order_id: str = Field(..., extra={'path_param': True}, description='The ID of the order to retrieve.')

    # Deprecated
    client_order_id: Optional[str] = Field(None, description='Deprecated: Client Order ID to fetch the order with.')
    user_native_currency: Optional[str] = Field(None, description='Deprecated: User native currency to fetch order '
                                                                  'with.')

    def endpoint(self) -> str:
        return f"orders/historical/{self.order_id}"


class CoinbaseAdvancedTradeListFillsRequest(
    _RequestGET,  # GET method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for ListFillsEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getfills
    ```json
    {
        "order_id": "string",
        "product_id": "string",
        "start_sequence_timestamp": "2021-01-01T00:00:00.123Z",
        "end_sequence_timestamp": "2021-01-01T00:00:00.123Z",
        "limit": 0,
        "cursor": "string"
    }
    ```
    """
    order_id: Optional[str] = Field(None, description='ID of order')
    product_id: Optional[str] = Field(None, description='The ID of the product this order was created for.')
    start_sequence_timestamp: Optional[UnixTimestampSecondFieldToDatetime] = Field(None,
                                                                                   description='Start date. '
                                                                                               'Only fills with '
                                                                                               'a trade time'
                                                                                               'at or after this '
                                                                                               'start date are '
                                                                                               'returned.')
    end_sequence_timestamp: Optional[UnixTimestampSecondFieldToDatetime] = Field(None,
                                                                                 description='End date. Only fills '
                                                                                             'with a trade time'
                                                                                             'before this start date '
                                                                                             'are returned.')
    limit: Optional[int] = Field(None, description='Maximum number of fills to return in response. Defaults to 100.')
    cursor: Optional[str] = Field(None, description='Cursor used for pagination. When provided, the response returns '
                                                    'responses after this cursor.')

    def endpoint(self) -> str:
        return "orders/historical/fills"

    class Config:
        json_encoders = {
            # TODO: Check on Coinbase Help for correct format
            datetime: lambda v: v.strftime("%Y-%m-%dT%H:%M:%S") + f".{v.microsecond // 1000:03d}Z",
        }


class CoinbaseAdvancedTradeGetProductBookRequest(
    _RequestGET,  # GET method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for ListProductsEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getproducts
    ```json
    {
        "product_id": "BTC-USD",
        "limit": 0
    }
    ```
    """
    product_id: str = Field(..., description='The trading pair to get book information for.')
    limit: Optional[int] = Field(None, description='Number of products to offset before returning.')

    def endpoint(self) -> str:
        return "product_book"


class CoinbaseAdvancedTradeGetBestBidAskRequest(
    _RequestGET,  # GET method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for ListProductsEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getproducts
    ```json
    {
        "product_ids":
        [
            "BTC-USD",
            "ETH-USD"
        ]
    }
    ```
    """
    product_ids: List[str] = Field(..., description='The trading pair to get book information for.')

    def endpoint(self) -> str:
        return "best_bid_ask"


class CoinbaseAdvancedTradeListProductsRequest(
    _RequestGET,  # GET method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for ListProductsEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getproducts
    ```json
    {
        "limit": 100,
        "offset": 0,
        "product_type": "SPOT"
    }
    ```
    """
    limit: Optional[int] = Field(None, description='A limit describing how many products to return.')
    offset: Optional[int] = Field(None, description='Number of products to offset before returning.')
    product_type: Optional[str] = Field(None, description='Type of products to return.')

    def endpoint(self) -> str:
        return "products"


class CoinbaseAdvancedTradeGetProductRequest(
    _RequestGET,  # GET method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for GetProductEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getproduct
    ```json
    {
        "product_id": "BTC-USD"
    }
    ```
    """
    product_id: str = Field(..., extra={'path_param': True}, description='The trading pair to get information for.')

    def endpoint(self) -> str:
        return f"products/{self.product_id}"


class CoinbaseAdvancedTradeGetProductCandlesRequest(
    _RequestGET,  # GET method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for GetProductCandlesEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getcandles
    ```json
    {
        "product_id": "BTC-USD",
        "start": "1577836800.0",
        "end": "1577923200.0",
        "granularity": "FIVE_MINUTE"
    }
    ```
    """
    product_id: str = Field(..., extra={'path_param': True}, description='The trading pair.')
    start: UnixTimestampSecondFieldToStr = Field(..., description='Timestamp for starting range of aggregations, '
                                                                  'in UNIX time.')
    end: UnixTimestampSecondFieldToStr = Field(..., description='Timestamp for ending range of aggregations, '
                                                                'in UNIX time.')
    granularity: str = Field(..., description='The time slice value for each candle.')

    class Config:
        json_encoders = {
            datetime: lambda v: str(v.timestamp()),
        }

    def endpoint(self) -> str:
        return f"products/{self.product_id}/candles"


class CoinbaseAdvancedTradeGetMarketTradesRequest(
    _RequestGET,  # GET method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for GetMarketTradesEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getmarkettrades
    ```json
    {
        "product_id": "BTC-USD",
        "limit": 100
    }
    ```
    """
    product_id: str = Field(..., extra={'path_param': True}, description="The trading pair, i.e., 'BTC-USD'.")
    limit: int = Field(..., description='Number of trades to return.')

    def endpoint(self) -> str:
        return f"products/{self.product_id}/ticker"


class CoinbaseAdvancedTradeGetTransactionSummaryRequest(
    _RequestGET,  # GET method settings
    PydanticMockableForJson,  # Generate samples from docstring JSON
    CoinbaseAdvancedTradeRequest,  # Sets the base type, registers the class
    CoinbaseAdvancedTradeEndpointRateLimit,  # Rate limit (Must be after CoinbaseAdvancedTradeRequest)
):
    """
    Dataclass representing request parameters for TransactionSummaryEndpoint.

    This is required for the test. It verifies that the request parameters are
    consistent with the Coinbase Advanced Trade API documentation.
    https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_gettransactionsummary
    ```json
    {
        "start_date": "2021-01-01T00:00:00.123Z",
        "end_date": "2021-01-02T00:00:00.123Z",
        "user_native_currency": "USD",
        "product_type": "SPOT"
    }
    ```
    """
    start_date: Optional[UnixTimestampSecondFieldToDatetime] = Field(None, description='Start date.')
    end_date: Optional[UnixTimestampSecondFieldToDatetime] = Field(None, description='End date.')
    user_native_currency: Optional[str] = Field(None, description='String of the users native currency, default is USD')
    product_type: Optional[str] = Field(None, description='Type of product')

    def endpoint(self) -> str:
        return "transaction_summary"

    class Config:
        json_encoders = {
            # TODO: Check on Coinbase Help for correct format
            datetime: lambda v: v.strftime("%Y-%m-%dT%H:%M:%S") + f".{v.microsecond // 1000:03d}Z",
        }
