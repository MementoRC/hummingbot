from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
import logging

from hummingbot.logger import HummingbotLogger


class RateSourceBase(ABC):
    _logger: HummingbotLogger | None = None

    @property
    @abstractmethod
    def name(self) -> str: ...

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    @abstractmethod
    async def get_prices(self, quote_token: str | None = None) -> dict[str, Decimal]: ...
