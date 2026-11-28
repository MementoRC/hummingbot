"""Protocols for executor factory type safety."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ExecutorConfigProtocol(Protocol):
    """Minimum config interface for factory registration."""

    id: str
    type: str
    controller_id: str | None
    connector_name: str
    trading_pair: str


@runtime_checkable
class ExecutorProtocol(Protocol):
    """Executor interface for factory creation."""

    config: ExecutorConfigProtocol

    def start(self) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class ExecutorUpdateProtocol(Protocol):
    """Protocol for executor update payloads."""

    def validate(self) -> bool: ...
