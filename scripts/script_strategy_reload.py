from hummingbot.core.clock import Clock
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class ScriptStrategyReload(ScriptStrategyBase):
    """
    Testing a reload method for Script Strategies
    """

    def on_tick(self):
        pass

    def stop(self, clock: Clock):
        self.logger().info(f"Stopping ScriptStrategyReload {self.script_module}")
        pass

    def format_status(self) -> str:
        """
        Returns status of the current strategy on user balances and current active orders. This function is called
        when status command is issued. Override this function to create custom status display output.
        """
        self.logger().info('format_status() called')
        if not self.ready_to_trade:
            return "Market connectors are not ready."
