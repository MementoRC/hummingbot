from decimal import Decimal
from unittest import TestCase
from unittest.mock import MagicMock

from hummingbot.logger import HummingbotLogger
from hummingbot.strategy_v2.utils.trailing_stop_manager import LadderedTrailingStop, TrailingStopManager


class TestTrailingStopManager(TestCase):
    def setUp(self) -> None:
        self.config = LadderedTrailingStop(
            activation_pnl_pct=Decimal("0.02"),
            trailing_pct=Decimal("0.01"),
            take_profit_table=((Decimal("0.03"), Decimal("0.5")), (Decimal("0.05"), Decimal("1"))),
        )
        # No relaxation keeps the trailing distance equal to trailing_pct.
        self.manager = TrailingStopManager(self.config, pnl_relaxation=Decimal("0"))
        self.on_close = MagicMock()
        self.on_partial = MagicMock()

    def _update(self, pnl: str, amount: str = "2") -> None:
        self.manager.update(Decimal(pnl), Decimal(amount), self.on_close, self.on_partial)

    def test_logger_is_cached(self):
        self.assertIsInstance(TrailingStopManager.logger(), HummingbotLogger)
        self.assertIs(TrailingStopManager.logger(), TrailingStopManager.logger())

    def test_initial_state_is_inactive(self):
        self.assertIsNone(self.manager.pnl_trigger)

    def test_update_rejects_non_positive_amount(self):
        with self.assertRaises(AssertionError):
            self._update("0.03", amount="0")
        with self.assertRaises(AssertionError):
            self._update("0.03", amount="-1")

    def test_not_activated_below_threshold(self):
        self._update("0.019")
        self.assertIsNone(self.manager.pnl_trigger)
        self.on_close.assert_not_called()
        self.on_partial.assert_not_called()

    def test_activation_at_threshold_sets_trigger_without_closing(self):
        self._update("0.02")
        self.assertEqual(self.manager.pnl_trigger, Decimal("0.01"))
        self.on_close.assert_not_called()
        self.on_partial.assert_not_called()

    def test_trigger_ratchets_up_only(self):
        self._update("0.02")
        self._update("0.04")
        self.assertEqual(self.manager.pnl_trigger, Decimal("0.03"))
        # A pullback that stays above the trigger must not lower it.
        self._update("0.035")
        self.assertEqual(self.manager.pnl_trigger, Decimal("0.03"))
        self.on_close.assert_not_called()
        self.on_partial.assert_not_called()

    def test_trigger_with_no_matching_ladder_step_closes_fully(self):
        self._update("0.02")
        self._update("0.01")
        self.on_close.assert_called_once_with()
        self.on_partial.assert_not_called()
        # Trigger is re-armed relative to the pnl that fired it.
        self.assertEqual(self.manager.pnl_trigger, Decimal("0.00"))

    def test_trigger_on_partial_ladder_step_closes_fraction(self):
        self._update("0.02")
        self._update("0.04")
        self._update("0.03", amount="2")
        self.on_partial.assert_called_once_with(Decimal("1.0"))
        self.on_close.assert_not_called()
        self.assertEqual(self.manager.pnl_trigger, Decimal("0.02"))

    def test_trigger_on_full_ladder_step_closes_fully(self):
        self._update("0.06")
        self.assertEqual(self.manager.pnl_trigger, Decimal("0.05"))
        self._update("0.05")
        self.on_close.assert_called_once_with()
        self.on_partial.assert_not_called()

    def test_trigger_edge_exactly_at_trigger_fires(self):
        self._update("0.03")
        self.assertEqual(self.manager.pnl_trigger, Decimal("0.02"))
        self._update("0.02")
        self.on_partial.assert_not_called()
        self.on_close.assert_called_once_with()

    def test_price_between_trigger_and_updated_trigger_does_nothing(self):
        self._update("0.02")
        self._update("0.015")
        self.assertEqual(self.manager.pnl_trigger, Decimal("0.01"))
        self.on_close.assert_not_called()
        self.on_partial.assert_not_called()

    def test_trailing_percentage_flat_when_pnl_not_above_base(self):
        self.assertEqual(self.manager._calculate_trailing_percentage(Decimal("0.01")), Decimal("0.01"))
        self.assertEqual(self.manager._calculate_trailing_percentage(Decimal("-0.5")), Decimal("0.01"))

    def test_trailing_percentage_relaxes_with_pnl(self):
        manager = TrailingStopManager(self.config, pnl_relaxation=Decimal("0.5"), max_trailing_pct=Decimal("0.5"))
        # base 0.01 + 0.04 * 0.5 = 0.03
        self.assertEqual(manager._calculate_trailing_percentage(Decimal("0.04")), Decimal("0.030"))

    def test_trailing_percentage_is_capped(self):
        manager = TrailingStopManager(self.config)
        self.assertEqual(manager._calculate_trailing_percentage(Decimal("0.5")), Decimal("0.05"))

    def test_activation_uses_relaxed_trailing_distance(self):
        manager = TrailingStopManager(self.config)  # relaxation 0.9, cap 0.05
        manager.update(Decimal("0.1"), Decimal("1"), self.on_close, self.on_partial)
        self.assertEqual(manager.pnl_trigger, Decimal("0.05"))

    def test_find_closest_take_profit(self):
        self.assertEqual(self.manager._find_closest_take_profit(Decimal("0.01")), (Decimal("0"), Decimal("1")))
        self.assertEqual(self.manager._find_closest_take_profit(Decimal("0.04")), (Decimal("0.03"), Decimal("0.5")))
        self.assertEqual(self.manager._find_closest_take_profit(Decimal("0.09")), (Decimal("0.05"), Decimal("1")))

    def test_empty_take_profit_table_always_closes_fully(self):
        config = LadderedTrailingStop(
            activation_pnl_pct=Decimal("0.02"), trailing_pct=Decimal("0.01"), take_profit_table=()
        )
        manager = TrailingStopManager(config, pnl_relaxation=Decimal("0"))
        manager.update(Decimal("0.03"), Decimal("1"), self.on_close, self.on_partial)
        manager.update(Decimal("0.02"), Decimal("1"), self.on_close, self.on_partial)
        self.on_close.assert_called_once_with()
        self.on_partial.assert_not_called()
