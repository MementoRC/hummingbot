"""
Exceptions used in the Hummingbot codebase.

Re-exported from the data-type-primitives sub-package (Phase 1 post-ADR-0001
extraction plan, hb-data-type-primitives#10).
"""

from data_type_primitives.exceptions import (
    ArgumentParserError,
    HummingbotBaseException,
    InvalidController,
    InvalidScriptModule,
    OracleRateUnavailable,
)

__all__ = [
    "ArgumentParserError",
    "HummingbotBaseException",
    "InvalidController",
    "InvalidScriptModule",
    "OracleRateUnavailable",
]
