from typing import TYPE_CHECKING

from hummingbot.connector.trading_rule import TradingRule

if TYPE_CHECKING:
    from data_type_primitives import TradingRules


def to_parent_trading_rule(rules: "TradingRules") -> TradingRule:
    """
    Convert a data_type_primitives TradingRules into the parent TradingRule.

    In the primitive, a zero ``min_*`` field and ``max_order_size=None`` mean "no constraint". The parent
    expresses that with its own defaults (a tiny increment / a huge maximum), so those fields are omitted
    from the constructor call rather than copied.
    """
    kwargs = {
        "min_order_size": rules.min_order_size,
        "min_notional_size": rules.min_notional_size,
        "supports_limit_orders": rules.supports_limit_orders,
        "supports_market_orders": rules.supports_market_orders,
    }
    if rules.max_order_size is not None:
        kwargs["max_order_size"] = rules.max_order_size
    if rules.min_price_increment > 0:
        kwargs["min_price_increment"] = rules.min_price_increment
    if rules.min_base_amount_increment > 0:
        kwargs["min_base_amount_increment"] = rules.min_base_amount_increment
    return TradingRule(trading_pair=rules.trading_pair, **kwargs)
