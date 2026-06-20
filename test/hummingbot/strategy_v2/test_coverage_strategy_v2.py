"""
Minimal coverage tests for strategy_v2 uncovered lines.
Targets: controller_base.py:373,377 | progressive_trading_controller.py:24-26,64-66,109
         strategy_v2_base.py:95,104,107-108,116,559,682
"""

from decimal import Decimal
from unittest.mock import MagicMock, mock_open, patch

# ---------------------------------------------------------------------------
# controller_base.py:373,377 — filter_executors close_timestamp range branches
# ---------------------------------------------------------------------------


def _make_executor(close_timestamp=None):
    e = MagicMock()
    e.close_timestamp = close_timestamp
    e.net_pnl_pct = Decimal("0.01")
    e.net_pnl_quote = Decimal("10")
    e.timestamp = 1000.0
    return e


def test_filter_executors_min_close_timestamp():
    """Line 373: min_close_timestamp filter list-comp."""
    from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase, ExecutorFilter

    config = ControllerConfigBase(id="test", controller_name="c", controller_type="generic")
    controller = ControllerBase.__new__(ControllerBase)
    controller.config = config

    e_with_ts = _make_executor(close_timestamp=2000.0)
    e_no_ts = _make_executor(close_timestamp=None)
    e_early = _make_executor(close_timestamp=500.0)

    ef = ExecutorFilter(min_close_timestamp=1000.0)
    # Pass executors directly to avoid needing self.executors_info initialised
    result = controller.filter_executors(executors=[e_with_ts, e_no_ts, e_early], executor_filter=ef)
    assert e_with_ts in result
    assert e_no_ts not in result
    assert e_early not in result


def test_filter_executors_max_close_timestamp():
    """Line 377: max_close_timestamp filter list-comp."""
    from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase, ExecutorFilter

    config = ControllerConfigBase(id="test", controller_name="c", controller_type="generic")
    controller = ControllerBase.__new__(ControllerBase)
    controller.config = config

    e_before = _make_executor(close_timestamp=500.0)
    e_after = _make_executor(close_timestamp=2000.0)
    e_no_ts = _make_executor(close_timestamp=None)

    ef = ExecutorFilter(max_close_timestamp=1000.0)
    result = controller.filter_executors(executors=[e_before, e_after, e_no_ts], executor_filter=ef)
    assert e_before in result
    assert e_after not in result
    assert e_no_ts not in result


# ---------------------------------------------------------------------------
# progressive_trading_controller.py:24-26 — coerce_manual_kill_switch(None)
# ---------------------------------------------------------------------------


def test_progressive_config_coerce_manual_kill_switch_none():
    """Lines 24-26: coerce_manual_kill_switch returns False when v is None."""
    from hummingbot.strategy_v2.controllers.progressive_trading_controller import ProgressiveTradingControllerConfig

    cfg = ProgressiveTradingControllerConfig(
        id="test",
        controller_name="progressive_trading",
        connector_name="binance",
        trading_pair="BTC-USDT",
        manual_kill_switch=None,
    )
    assert cfg.manual_kill_switch is False


# ---------------------------------------------------------------------------
# progressive_trading_controller.py:64-66 — validate_apr_yield branches
# ---------------------------------------------------------------------------


def test_progressive_config_validate_apr_yield_empty_string():
    """Line 65 (empty str branch): validate_apr_yield returns None for ''."""
    from hummingbot.strategy_v2.controllers.progressive_trading_controller import ProgressiveTradingControllerConfig

    cfg = ProgressiveTradingControllerConfig(
        id="test",
        controller_name="progressive_trading",
        connector_name="binance",
        trading_pair="BTC-USDT",
        apr_yield="",
    )
    assert cfg.apr_yield is None


def test_progressive_config_validate_apr_yield_string_value():
    """Line 65 (non-empty str branch): validate_apr_yield coerces string to Decimal."""
    from hummingbot.strategy_v2.controllers.progressive_trading_controller import ProgressiveTradingControllerConfig

    cfg = ProgressiveTradingControllerConfig(
        id="test",
        controller_name="progressive_trading",
        connector_name="binance",
        trading_pair="BTC-USDT",
        apr_yield="0.25",
    )
    assert cfg.apr_yield == Decimal("0.25")


# ---------------------------------------------------------------------------
# progressive_trading_controller.py:109 — to_format_status with non-empty df
# ---------------------------------------------------------------------------


def test_progressive_controller_to_format_status_non_empty():
    """Line 109: to_format_status returns formatted string when df is non-empty."""
    import pandas as pd

    from hummingbot.strategy_v2.controllers.progressive_trading_controller import (
        ProgressiveTradingController,
        ProgressiveTradingControllerConfig,
    )

    cfg = ProgressiveTradingControllerConfig(
        id="test",
        controller_name="progressive_trading",
        connector_name="binance",
        trading_pair="BTC-USDT",
    )
    controller = ProgressiveTradingController.__new__(ProgressiveTradingController)
    controller.config = cfg
    controller.processed_data = {"features": pd.DataFrame({"col": [1, 2, 3]})}

    result = controller.to_format_status()
    assert isinstance(result, list)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# strategy_v2_base.py:95 — parse_controllers_config non-empty string
# ---------------------------------------------------------------------------


def test_strategy_v2_config_parse_controllers_config_non_empty_string():
    """Line 95: non-empty string splits into list of paths."""
    from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase

    cfg = StrategyV2ConfigBase(controllers_config="config_a.yml, config_b.yml")
    assert cfg.controllers_config == ["config_a.yml", "config_b.yml"]


# ---------------------------------------------------------------------------
# strategy_v2_base.py:104,107-108,116 — load_controller_configs
# ---------------------------------------------------------------------------


def test_strategy_v2_config_load_controller_configs():
    """Lines 104,107-108,116: load_controller_configs reads file and imports module."""
    import types

    from hummingbot.strategy.strategy_v2_base import StrategyV2ConfigBase

    fake_yaml = {"controller_type": "generic", "controller_name": "my_ctrl", "id": "test"}

    # Build a fake config class the finder will match.
    # FakeConfig must be resolvable via issubclass against the *same* ControllerConfigBase
    # that strategy_v2_base.py imports — use that module's class directly.
    import hummingbot.strategy.strategy_v2_base as _sv2_mod

    _ControllerConfigBase = _sv2_mod.ControllerConfigBase

    class FakeConfig(_ControllerConfigBase):
        pass

    # Use a real module object so inspect.getmembers / inspect.isclass work naturally
    fake_module = types.ModuleType("controllers.generic.my_ctrl")
    fake_module.FakeConfig = FakeConfig

    with (
        patch("hummingbot.strategy.strategy_v2_base.settings.CONTROLLERS_CONF_DIR_PATH", "/fake/conf"),
        patch("hummingbot.strategy.strategy_v2_base.settings.CONTROLLERS_MODULE", "controllers"),
        patch("builtins.open", mock_open(read_data="")),
        patch("hummingbot.strategy.strategy_v2_base.yaml.safe_load", return_value=fake_yaml),
        patch("hummingbot.strategy.strategy_v2_base.importlib.import_module", return_value=fake_module),
    ):
        cfg = StrategyV2ConfigBase(controllers_config=["my_ctrl.yml"])
        result = cfg.load_controller_configs()

    assert len(result) == 1
    assert isinstance(result[0], FakeConfig)


# ---------------------------------------------------------------------------
# strategy_v2_base.py:559 — format_status positions loop (positions_data.append)
# ---------------------------------------------------------------------------


def test_strategy_v2_base_format_status_with_positions():
    """Line 559: positions_data.append executed when positions list is non-empty."""
    import pandas as pd

    from hummingbot.core.data_type.common import TradeType
    from hummingbot.strategy.strategy_v2_base import StrategyV2Base
    from hummingbot.strategy_v2.executors.data_types import PositionSummary

    strategy = StrategyV2Base.__new__(StrategyV2Base)
    strategy.connectors = {}
    strategy._set_current_timestamp(1000.0)
    strategy.ready_to_trade = True

    pos = PositionSummary(
        connector_name="binance",
        trading_pair="BTC-USDT",
        volume_traded_quote=Decimal("1000"),
        side=TradeType.BUY,
        amount=Decimal("0.1"),
        breakeven_price=Decimal("50000"),
        unrealized_pnl_quote=Decimal("10"),
        realized_pnl_quote=Decimal("5"),
        cum_fees_quote=Decimal("1"),
    )

    mock_controller = MagicMock()
    mock_controller.to_format_status.return_value = []
    strategy.controllers = {"ctrl1": mock_controller}
    strategy.controller_reports = {"ctrl1": {"positions": [pos]}}

    with (
        patch.object(strategy, "get_executors_by_controller", return_value=[]),
        patch.object(strategy, "get_positions_by_controller", return_value=[pos]),
        patch.object(strategy, "network_warning", return_value=[]),
        patch.object(strategy, "get_market_trading_pair_tuples", return_value=[]),
        patch.object(strategy, "get_balance_df", return_value=pd.DataFrame({"Asset": [], "Total": []})),
        patch.object(strategy, "active_orders_df", side_effect=ValueError),
        patch.object(strategy, "executors_info_to_df", return_value=pd.DataFrame()),
    ):
        result = strategy.format_status()

    assert result is not None
    assert "BTC-USDT" in result


# ---------------------------------------------------------------------------
# strategy_v2_base.py:682 — _collect_initial_positions with initial_positions
# ---------------------------------------------------------------------------


def test_strategy_v2_base_collect_initial_positions():
    """Line 682: initial_positions_by_controller populated when controller has positions."""
    from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase

    strategy = StrategyV2Base.__new__(StrategyV2Base)

    mock_config = MagicMock(spec=StrategyV2ConfigBase)
    strategy.config = mock_config

    fake_position = MagicMock()
    mock_controller_cfg = MagicMock()
    mock_controller_cfg.id = "ctrl1"
    mock_controller_cfg.initial_positions = [fake_position]

    mock_config.load_controller_configs.return_value = [mock_controller_cfg]

    result = strategy._collect_initial_positions()

    assert "ctrl1" in result
    assert result["ctrl1"] == [fake_position]
