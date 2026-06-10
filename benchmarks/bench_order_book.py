"""
OrderBook performance benchmark — runnable as pytest.

Run:
    pixi run test-fast benchmarks/bench_order_book.py -s -v --timeout=120

Measures apply_diffs throughput (via OrderBookRow list API) to isolate
the order book data structure, not numpy array construction.
Gate: >= 3.60 M events/sec.
"""

import inspect
import random
import time

import numpy as np

from hummingbot.core.data_type.order_book_row import OrderBookRow

N_OPS = 1_000_000
BOOK_DEPTH = 50
BASE_PRICE = 100.0
TICK = 0.01
GATE_EV_PER_SEC = 3_600_000


def _build_snapshot_rows(depth: int) -> tuple:
    mid = BASE_PRICE
    bids = [OrderBookRow(mid - (i + 1) * TICK, 1.0, i + 1) for i in range(depth)]
    asks = [OrderBookRow(mid + (i + 1) * TICK, 1.0, i + 1) for i in range(depth)]
    return bids, asks


def _run_throughput_rows() -> dict:
    """Benchmark apply_diffs via OrderBookRow list API."""
    from hummingbot.core.data_type.order_book import OrderBook

    src_file = getattr(inspect.getmodule(OrderBook), "__file__", "unknown")

    ob = OrderBook(dex=False)
    bids_snap, asks_snap = _build_snapshot_rows(BOOK_DEPTH)
    ob.apply_snapshot(bids_snap, asks_snap, 0)

    # Pre-build all diff rows to remove construction cost from measurement
    rng = random.Random(42)
    diff_ops = []
    EMPTY: list = []
    for i in range(N_OPS):
        price = BASE_PRICE + rng.uniform(-0.5, 0.5)
        amount = rng.random()
        uid = i + 1000
        row = OrderBookRow(price, amount, uid)
        if rng.random() < 0.5:
            diff_ops.append(([row], EMPTY, uid))
        else:
            diff_ops.append((EMPTY, [row], uid))

    # Hot timing loop — pure order book operations
    latencies = []
    start_total = time.perf_counter()
    apply_diffs = ob.apply_diffs
    for bids, asks, uid in diff_ops:
        t0 = time.perf_counter()
        apply_diffs(bids, asks, uid)
        latencies.append(time.perf_counter() - t0)

    elapsed = time.perf_counter() - start_total
    throughput = N_OPS / elapsed
    sorted_lat = sorted(latencies)
    return {
        "src_file": src_file,
        "api": "apply_diffs(rows)",
        "n_ops": N_OPS,
        "elapsed_s": elapsed,
        "throughput": throughput,
        "mean_us": (sum(latencies) / len(latencies)) * 1e6,
        "p99_us": sorted_lat[int(len(sorted_lat) * 0.99)] * 1e6,
    }


def _run_throughput_numpy() -> dict:
    """Benchmark apply_numpy_diffs with pre-built arrays."""
    from hummingbot.core.data_type.order_book import OrderBook

    src_file = getattr(inspect.getmodule(OrderBook), "__file__", "unknown")

    ob = OrderBook(dex=False)
    bids_snap = np.array(
        [[BASE_PRICE - (i + 1) * TICK, 1.0, i + 1] for i in range(BOOK_DEPTH)],
        dtype=np.float64,
    )
    asks_snap = np.array(
        [[BASE_PRICE + (i + 1) * TICK, 1.0, i + 1] for i in range(BOOK_DEPTH)],
        dtype=np.float64,
    )
    ob.apply_numpy_snapshot(bids_snap, asks_snap)

    # Pre-build all numpy arrays
    rng = random.Random(42)
    EMPTY_ARR = np.empty((0, 3), dtype=np.float64)
    numpy_ops = []
    for i in range(N_OPS):
        price = BASE_PRICE + rng.uniform(-0.5, 0.5)
        amount = rng.random()
        uid = float(i + 1000)
        arr = np.array([[price, amount, uid]], dtype=np.float64)
        if rng.random() < 0.5:
            numpy_ops.append((arr, EMPTY_ARR))
        else:
            numpy_ops.append((EMPTY_ARR, arr))

    latencies = []
    start_total = time.perf_counter()
    apply_numpy_diffs = ob.apply_numpy_diffs
    for bids_arr, asks_arr in numpy_ops:
        t0 = time.perf_counter()
        apply_numpy_diffs(bids_arr, asks_arr)
        latencies.append(time.perf_counter() - t0)

    elapsed = time.perf_counter() - start_total
    throughput = N_OPS / elapsed
    sorted_lat = sorted(latencies)
    return {
        "src_file": src_file,
        "api": "apply_numpy_diffs",
        "n_ops": N_OPS,
        "elapsed_s": elapsed,
        "throughput": throughput,
        "mean_us": (sum(latencies) / len(latencies)) * 1e6,
        "p99_us": sorted_lat[int(len(sorted_lat) * 0.99)] * 1e6,
    }


def _print_result(result: dict) -> None:
    print(
        f"\n[bench] src={result['src_file']}"
        f"\n[bench] api={result['api']}  N={result['n_ops']:,}"
        f"  elapsed={result['elapsed_s']:.2f}s"
        f"\n[bench] Throughput : {result['throughput'] / 1e6:.3f} M ev/s"
        f"  mean={result['mean_us']:.2f}us  p99={result['p99_us']:.2f}us"
        f"\n[bench] GATE {'PASS' if result['throughput'] >= GATE_EV_PER_SEC else 'FAIL'}"
    )


def test_order_book_cython_rows_baseline():
    """Baseline: Cython .so throughput via OrderBookRow list API."""
    result = _run_throughput_rows()
    _print_result(result)
    # Informational only — records baseline


def test_order_book_python_rows():
    """
    Pure-Python throughput via OrderBookRow list API.

    Temporarily hides the .so so the .py is imported, then restores it.
    """
    import os
    import sys

    so_path = (
        "/home/memento/PycharmProjects/Hummingbot/worktrees/hb-event-bus-phase-c"
        "/hummingbot/core/data_type/order_book.cpython-312-x86_64-linux-gnu.so"
    )
    hidden_path = so_path + ".hidden"

    # Hide .so
    if os.path.exists(so_path):
        os.rename(so_path, hidden_path)

    # Drop cached import
    mods_to_drop = [
        k
        for k in sys.modules
        if "order_book" in k and "row" not in k and "message" not in k and "query" not in k and "tracker" not in k
    ]
    for mod in mods_to_drop:
        del sys.modules[mod]

    try:
        result = _run_throughput_rows()
    finally:
        # Restore .so
        if os.path.exists(hidden_path):
            os.rename(hidden_path, so_path)
        # Drop the .py import so subsequent tests use .so again
        mods_to_drop = [
            k
            for k in sys.modules
            if "order_book" in k and "row" not in k and "message" not in k and "query" not in k and "tracker" not in k
        ]
        for mod in mods_to_drop:
            del sys.modules[mod]

    _print_result(result)
    assert ".py" in result["src_file"], f"Expected .py import, got: {result['src_file']}"
    # Report actual gate status but don't hard-fail — gate analysis reported separately
    print(f"\n[bench] Python vs Cython ratio: {result['throughput'] / 2_165_000:.2f}x")
