"""Pure-Python port of transaction_tracker.pyx.

TransactionTracker monitors in-flight transactions and fires timeout
callbacks when they exceed their allotted time window.
"""

from __future__ import annotations

from hummingbot.core.time_iterator import TimeIterator


class TransactionTracker(TimeIterator):
    def __init__(self) -> None:
        super().__init__()
        self._tx_time_limits: dict[str, float] = {}

    def tick(self, timestamp: float) -> None:
        super().tick(timestamp)
        self.c_process_tx_timeouts()

    def c_start_tx_tracking(self, tx_id: str, timeout_seconds: float) -> None:
        if tx_id in self._tx_time_limits:
            raise ValueError(f"The transaction {tx_id} is already being monitored.")
        self._tx_time_limits[tx_id] = self._current_timestamp + timeout_seconds

    def c_stop_tx_tracking(self, tx_id: str) -> None:
        if tx_id not in self._tx_time_limits:
            return
        del self._tx_time_limits[tx_id]

    def c_is_tx_tracked(self, tx_id: str) -> bool:
        return tx_id in self._tx_time_limits

    def c_did_timeout_tx(self, tx_id: str) -> None:
        self.c_stop_tx_tracking(tx_id)

    def c_process_tx_timeouts(self) -> None:
        timed_out_tx_ids: list[str] = []
        for tx_id, time_limit in self._tx_time_limits.items():
            if self._current_timestamp > time_limit:
                timed_out_tx_ids.append(tx_id)
        for tx_id in timed_out_tx_ids:
            self.c_did_timeout_tx(tx_id)
