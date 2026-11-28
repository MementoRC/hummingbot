"""Executor factory with decorator-based registration.

Replaces the string-keyed _executor_mapping dict on ExecutorOrchestrator
with a type-keyed registry that executors self-register into at import time.

Usage:
    @ExecutorFactory.register(PositionExecutorConfig)
    class PositionExecutor(ExecutorBase):
        ...

    # Then in orchestrator:
    executor = ExecutorFactory.create(strategy, config, update_interval, max_retries)
"""

from __future__ import annotations

from collections.abc import Callable
import logging

from hummingbot.strategy_v2.executors.data_types import ExecutorConfigBase


class ExecutorFactory:
    """Registry-based factory for executor creation.

    Executors register themselves via the @register decorator,
    keyed by their config class type. Lookup is O(1) dict access.
    """

    _registry: dict[type[ExecutorConfigBase], type] = {}
    _logger = None

    @classmethod
    def logger(cls):
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    @classmethod
    def register(
        cls,
        config_type: type[ExecutorConfigBase],
    ) -> Callable:
        """Decorator to register an executor class for a config type.

        Usage:
            @ExecutorFactory.register(MyExecutorConfig)
            class MyExecutor(ExecutorBase):
                ...
        """

        def decorator(executor_cls):
            if config_type in cls._registry:
                cls.logger().warning(
                    f"Overriding executor registration for {config_type.__name__}: "
                    f"{cls._registry[config_type].__name__} -> {executor_cls.__name__}"
                )
            cls._registry[config_type] = executor_cls
            return executor_cls

        return decorator

    @classmethod
    def create(
        cls,
        strategy,
        config: ExecutorConfigBase,
        update_interval: float = 1.0,
        max_retries: int = 10,
    ):
        """Create an executor instance from a config object.

        Looks up the executor class registered for the config's type,
        then instantiates it with the standard constructor signature.

        Args:
            strategy: The StrategyV2Base instance.
            config: The executor configuration.
            update_interval: Tick interval in seconds.
            max_retries: Maximum retry attempts for failed orders.

        Returns:
            An executor instance.

        Raises:
            ValueError: If no executor is registered for the config type.
        """
        executor_cls = cls._registry.get(type(config))
        if executor_cls is None:
            raise ValueError(
                f"No executor registered for config type: {type(config).__name__}. "
                f"Registered types: {[t.__name__ for t in cls._registry]}"
            )
        return executor_cls(
            strategy=strategy,
            config=config,
            update_interval=update_interval,
            max_retries=max_retries,
        )

    @classmethod
    def get_registry(cls) -> dict[type[ExecutorConfigBase], type]:
        """Return the current registry (for introspection/debugging)."""
        return dict(cls._registry)

    @classmethod
    def is_registered(cls, config_type: type[ExecutorConfigBase]) -> bool:
        """Check if a config type has a registered executor."""
        return config_type in cls._registry
