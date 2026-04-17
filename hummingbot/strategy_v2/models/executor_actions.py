from typing import Any, TypeVar

from pydantic import BaseModel

from hummingbot.strategy_v2.executors.data_types import ExecutorConfigBase

ExecutorConfigType = TypeVar("ExecutorConfigType", bound=ExecutorConfigBase)


class ExecutorAction(BaseModel):
    """
    Base class for bot actions.
    """

    controller_id: str | None = "main"


class CreateExecutorAction(ExecutorAction):
    """
    Action to create an executor.
    """

    executor_config: ExecutorConfigType


class StopExecutorAction(ExecutorAction):
    """
    Action to stop an executor.
    """

    executor_id: str
    keep_position: bool | None = False


class StoreExecutorAction(ExecutorAction):
    """
    Action to store an executor.
    """

    executor_id: str


class UpdateExecutorAction(ExecutorAction):
    """
    Action to update a running executor with new data (e.g., volatility).
    """
    executor_id: str
    update_data: Any
