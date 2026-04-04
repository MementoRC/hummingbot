---
name: cython-transition
context: fork
model: claude-sonnet-4-6
description: "Guide Cython-to-Augmented-Pure-Python migration for hummingbot modules. Covers pragma conventions, annotation reference, triple-layout pattern, common pitfalls, hb-cython-framework integration, and hatch-cython build config. Use when converting .pyx files, adding Cython annotations to .py files, setting up augmented pure Python modules, or debugging Cython compilation issues."
allowed-tools:
  - Read
  - Grep
  - Glob
  - WebFetch
  - mcp__git__discover_tools
  - mcp__git__get_tool_spec
  - mcp__git__execute_tool
---

# Cython-to-Augmented Pure Python Transition Guide

## Purpose

Guide the progressive migration of hummingbot's Cython `.pyx` files to Augmented Pure Python — `.py` files with `cython.*` annotations that run as plain Python AND compile to native C extensions via Cython.

**Reference documentation:** https://cython.readthedocs.io/en/latest/src/tutorial/pure.html

---

## 1. Fetch Latest Cython Pure Python Documentation

Before advising on any conversion, fetch the current Cython pure Python mode docs:

```
WebFetch: https://cython.readthedocs.io/en/latest/src/tutorial/pure.html
```

Key sections to review:
- "Augmented mode" — how `.py` files with pragmas are compiled
- `@cython.ccall`, `@cython.cfunc` — the two function declaration modes
- `cython.declare()` — typed variable declarations
- `@cython.locals()` — function-local type annotations
- Typed memoryviews — `cython.double[:]` syntax

---

## 2. Our Framework: hb-cython-framework

The tooling lives in `sub-packages/cython-framework/` (repo: MementoRC/hb-cython-framework):

| Component | Location | Purpose |
|-----------|----------|---------|
| Test framework | `cython_framework/testing/cython_test_case.py` | `CythonTestCase`, `@cython_test_implementations()` — tests 4 variants |
| Pre-commit hook | `cython_framework/hooks/link_augmented_pyx.py` | Creates `.pyx` symlinks for augmented `.py` files |
| Build config | `pyproject.toml` `[tool.hatch.build.targets.wheel.hooks.cython]` | hatch-cython >=0.6.0 compilation |
| Compilation tests | `tests/unit/test_cython_compilation.py` | Proves full `.py` → `.pyx` → `.c` → `.so` → import pipeline |

Sub-packages add `hb-cython-framework` as a dev dependency to use the test framework.

---

## 3. The Augmented Pure Python Pattern

### File Header (Required Pragmas)

```python
# cython: language_level=3str
# cython: augmented_pure_python=True
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
```

Optional optimization pragmas (add per-file or per-function):
```python
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
```

### Directory Layout (Triple Pattern)

```
module/
  ├── my_module.py                # Augmented (runs as Python, compiles to C)
  ├── __pure_python__/
  │   └── my_module.py            # Canonical pure Python (no cython imports)
  └── __pure_cython__/            # Optional: standalone .pyx rewrite
      ├── my_module.pyx
      └── my_module.pxd
```

The augmented `.py` is the PRIMARY source. `__pure_python__/` is a clean fallback for tooling (mypy, IDE). `__pure_cython__/` is optional for maximum-performance rewrites.

---

## 4. Cython Annotation Reference

### Function Decorators

| Decorator | Compiles To | Python-Callable? | Use When |
|-----------|-------------|-------------------|----------|
| `@cython.ccall` | `cpdef` | YES | **Default choice** — callable from Python and C |
| `@cython.cfunc` | `cdef` | NO | Internal helpers only — tests/app code CANNOT call these |
| `@cython.nogil` | releases GIL | N/A | Pure C loops with no Python objects |
| `@cython.inline` | inlined | N/A | Small hot functions |

**CRITICAL: Always use `@cython.ccall` for functions that tests or application code calls.**
`@cython.cfunc` makes the function invisible to Python — only other Cython code can call it.

### Type Annotations

```python
import cython

# Typed variables
x: cython.int = 0
y: cython.double = 0.0
n: cython.size_t = 0

# Typed memoryviews (contiguous C arrays)
data: cython.double[:] = np.array([1.0, 2.0], dtype=np.float64)

# Function-level locals
@cython.locals(i=cython.int, total=cython.double)
def compute(data: cython.double[:]) -> cython.double:
    total = 0.0
    for i in range(len(data)):
        total += data[i]
    return total

# Module-level declarations
MAX_SIZE = cython.declare(cython.int, 1024)
```

### Optimization Decorators (Stack on Functions)

```python
@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.exceptval(check=False)  # skip exception propagation overhead
def hot_loop(data: cython.double[:], n: cython.int) -> cython.double:
    ...
```

### GIL Management

```python
@cython.cfunc
@cython.nogil  # entire function runs without GIL
def pure_c_work(data: cython.double[:], n: cython.int) -> cython.double:
    # No Python objects allowed here
    total: cython.double = 0.0
    for i in range(n):
        total += data[i]
    return total

@cython.ccall
@cython.nogil
def mixed_work(data: cython.double[:], n: cython.int):
    result: cython.double = pure_c_work(data, n)
    with cython.gil:  # re-acquire GIL for Python operations
        return np.array([result])
```

---

## 5. Conversion Checklist: .pyx → Augmented .py

### Step 1: Assess the .pyx File
- [ ] Identify all `cdef`/`cpdef` functions → map to `@cython.cfunc`/`@cython.ccall`
- [ ] Identify `cdef class` → `@cython.cclass` + `@dataclass` if appropriate
- [ ] Identify C-typed local variables → `@cython.locals()` or inline annotations
- [ ] Identify `.pxd` declarations → most are NOT needed (pragma header replaces them)
- [ ] Check for `cimport` statements → convert to regular `import` + cython annotations
- [ ] Check for `nogil` blocks → `@cython.nogil` decorator or `with cython.nogil:`

### Step 2: Create the Augmented .py
- [ ] Add pragma header (language_level, augmented_pure_python, distutils)
- [ ] `import cython` at top
- [ ] Convert `cdef`/`cpdef` → decorators (`@cython.cfunc`/`@cython.ccall`)
- [ ] Convert C type declarations → `cython.int`, `cython.double`, `cython.double[:]`
- [ ] Convert `cdef class` → `@cython.cclass`
- [ ] Verify file runs as plain Python (`python my_module.py` with no errors)

### Step 3: Create __pure_python__/ Fallback
- [ ] Copy augmented file, remove ALL `import cython` and `@cython.*` decorators
- [ ] Replace `cython.double[:]` → `list[float]` or `np.ndarray`
- [ ] Replace `cython.int` → `int`, `cython.double` → `float`
- [ ] Verify identical behavior: same function signatures, same results

### Step 4: Create .pyx Symlink
- [ ] Run `scripts/link_augmented_pyx.py` or manually: `ln -s my_module.py my_module.pyx`
- [ ] Verify: `ls -la my_module.pyx` shows symlink to `.py`

### Step 5: Test All Variants
- [ ] Write test class extending `CythonTestCase` from `cython_framework.testing`
- [ ] Set `MODULE_PATH` and `MODULE_NAME`
- [ ] Use `@cython_test_implementations()` decorator on each test
- [ ] Run: all variants produce identical results

### Step 6: Build and Verify Compilation
- [ ] `python -m cython --3str my_module.pyx` → produces `.c` file
- [ ] `python -m build --wheel` (with hatch-cython) → produces `.so` in wheel
- [ ] Import compiled module → `cython.compiled` is True

---

## 6. Common Pitfalls

| # | Pitfall | Fix |
|---|---------|-----|
| 1 | `@classmethod` + `@cython.cfunc` | Use `@cython.ccall` — cfunc is C-only, classmethod needs Python |
| 2 | `InitVar` vs stored field in dataclass | Augmented uses regular field (becomes C struct member); pure Python uses `InitVar` for init-only |
| 3 | `assert` stripped by `python -O` | Use explicit `if not x: raise ValueError()` for runtime guards |
| 4 | `to_float_array` returning `list` but annotated `cython.double[:]` | Memoryview and list are different — pick one and be consistent |
| 5 | `globals().update()` for module-level constants | Fragile — use explicit `__all__` instead |
| 6 | `.so.bak` rename for testing without compilation | Use `importlib.util.spec_from_file_location()` with explicit `.py` path |
| 7 | `sys.modules` key collision between variants | Namespace keys: `f"{module_path}.__{variant}__.{module_name}"` |
| 8 | `.pxd` files alongside augmented `.py` | NOT needed — pragma header is sufficient for augmented mode |

---

## 7. Build Integration (hatch-cython)

In `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.18.0", "hatch-cython>=0.6.0", "numpy>=1.20.0"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel.hooks.cython]
dependencies = ["hatch-cython>=0.6.0", "numpy>=1.20.0"]

[tool.hatch.build.targets.wheel.hooks.cython.options]
compile_py = false          # Only compile .pyx files (symlinked from augmented .py)
include_numpy = true
directives = { boundscheck = false, nonecheck = false, language_level = 3, binding = true }
define_macros = [["NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"]]
```

The `.pyx` symlinks (created by pre-commit hook) are what hatch-cython discovers and compiles.

---

## 8. Reference Branches

| Branch | Content | Status |
|--------|---------|--------|
| `dev/cython_candles` | Complete Coinbase candles in triple-layout | Reference for hb-candles-feed |
| `dev/cython-coinbase-candles` | Vectorized metrics with `@cython.nogil` | Extracted to market-simulator plan |
| `_for_bleed/augmented-pure-python` | Framework integration into bleeding-edge | Merged |

---

## 9. Migration Priority

1. **Performance metrics** (drawdown, Sharpe, profit-factor) — tight numerical loops
2. **Candles data types** (CandleData, utils) — data-intensive processing
3. **ConnectorBase / ExchangeBase** — core infrastructure, biggest impact
4. **strategy_base / order_tracker** — strategy execution hot path
