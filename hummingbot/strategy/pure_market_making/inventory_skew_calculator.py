"""Pure Python replacement for the Cython inventory_skew_calculator.pyx.

Retained because liquidity_mining depends on calculate_bid_ask_ratios_from_base_asset_ratio.
The PureMarketMakingStrategy itself is out-of-scope and was removed.
"""

import numpy as np

from .data_types import InventorySkewBidAskRatios


def calculate_bid_ask_ratios_from_base_asset_ratio(
    base_asset_amount: float,
    quote_asset_amount: float,
    price: float,
    target_base_asset_ratio: float,
    base_asset_range: float,
) -> InventorySkewBidAskRatios:
    total_portfolio_value = base_asset_amount * price + quote_asset_amount

    if total_portfolio_value <= 0.0 or base_asset_range <= 0.0:
        return InventorySkewBidAskRatios(0.0, 0.0)

    base_asset_value = base_asset_amount * price
    base_asset_range_value = min(base_asset_range * price, total_portfolio_value * 0.5)
    target_base_asset_value = total_portfolio_value * target_base_asset_ratio
    left_base_asset_value_limit = max(target_base_asset_value - base_asset_range_value, 0.0)
    right_base_asset_value_limit = target_base_asset_value + base_asset_range_value

    if base_asset_value < target_base_asset_value:
        left_inventory_ratio = float(
            np.interp(base_asset_value, [left_base_asset_value_limit, target_base_asset_value], [0.0, 0.5])
        )
        bid_adjustment = float(np.interp(left_inventory_ratio, [0, 0.5], [2.0, 1.0]))
    else:
        right_inventory_ratio = float(
            np.interp(base_asset_value, [target_base_asset_value, right_base_asset_value_limit], [0.5, 1.0])
        )
        bid_adjustment = float(np.interp(right_inventory_ratio, [0.5, 1], [1.0, 0.0]))

    ask_adjustment = 2.0 - bid_adjustment
    return InventorySkewBidAskRatios(bid_adjustment, ask_adjustment)
