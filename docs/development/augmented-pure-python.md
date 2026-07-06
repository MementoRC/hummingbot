# Augmented Pure Python Migration Guide

## Overview

Hummingbot uses Cython extensively for performance-critical components (ConnectorBase, strategy_base, order_tracker). The Augmented Pure Python approach migrates `.pyx` files to `.py` files with `cython.*` annotations that:

- **Run as plain Python** -- no compilation needed for development/testing
- **Compile to native C extensions** -- via Cython for production performance
- **Maintain a clean fallback** -- `__pure_python__/` directory with zero cython imports

## Quick Start

### 1. Install the framework

The `hb-cython-framework` sub-package provides all tooling:
- Test framework: `CythonTestCase`, `@cython_test_implementations()`
- Pre-commit hook: automatic `.pyx` symlink management
- Build config: `hatch-cython` integration for wheel compilation

### 2. Use the Claude Code skill

Invoke `/cython-transition` in any Claude Code session for guided conversion assistance.

### 3. Use the agents

- `focused-cython-converter` -- performs the actual `.pyx` -> augmented `.py` conversion
- `focused-cython-validator` -- validates correctness of augmented files

## The Triple-Layout Pattern

```
module/
+-- my_module.py                 # Augmented (primary source)
+-- __pure_python__/
|   +-- my_module.py             # Clean Python fallback
+-- __pure_cython__/             # Optional: standalone .pyx rewrite
    +-- my_module.pyx
    +-- my_module.pxd
```

### Augmented File Header

Every augmented `.py` file MUST start with these pragmas:

```python
# cython: language_level=3str
# cython: augmented_pure_python=True
# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
```

Optional optimization pragmas (file-level or per-function):
```python
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
```

## Annotation Reference

### Function Decorators

| Cython (.pyx) | Augmented (.py) | Python-Callable? | Use When |
|----------------|-----------------|-------------------|----------|
| `cdef func()` | `@cython.cfunc` | **NO** | Internal C helpers only |
| `cpdef func()` | `@cython.ccall` | **YES** | **Default choice** -- public functions |
| `cdef class` | `@cython.cclass` | YES | Cython extension types |

**CRITICAL**: Always use `@cython.ccall` for functions that tests or application code calls. `@cython.cfunc` makes the function invisible to Python.

### Type Annotations

| Cython (.pyx) | Augmented (.py) |
|----------------|-----------------|
| `cdef int x` | `x: cython.int` |
| `cdef double y` | `y: cython.double` |
| `cdef double[:] arr` | `arr: cython.double[:]` |
| `cdef size_t n` | `n: cython.size_t` |

### GIL Management

```python
@cython.ccall
@cython.nogil  # releases GIL -- pure C, no Python objects
def fast_compute(data: cython.double[:], n: cython.int) -> cython.double:
    total: cython.double = 0.0
    for i in range(n):
        total += data[i]
    return total
```

Re-acquire GIL when needed:
```python
with cython.gil:
    result_array = np.array([total])  # Python object allocation
```

## Conversion Checklist

1. [ ] Read the source `.pyx` file -- identify all cdef/cpdef, typed vars, cimports
2. [ ] Create augmented `.py` -- add pragma header, convert annotations
3. [ ] Create `__pure_python__/` fallback -- strip all cython references
4. [ ] Create `.pyx` symlink -- `ln -s my_module.py my_module.pyx`
5. [ ] Write tests -- `CythonTestCase` with `@cython_test_implementations()`
6. [ ] Verify -- both variants produce identical results
7. [ ] Compile -- `python -m cython --3str my_module.pyx` succeeds

## Common Pitfalls

| # | Pitfall | Fix |
|---|---------|-----|
| 1 | `@classmethod` + `@cython.cfunc` | Use `@cython.ccall` |
| 2 | `InitVar` vs stored field in dataclass | Be consistent between variants |
| 3 | `assert` stripped by `python -O` | Use `if not x: raise ValueError()` |
| 4 | Return type mismatch (`cython.double[:]` vs `list`) | Pick one, be consistent |
| 5 | `globals().update()` for constants | Use explicit `__all__` |
| 6 | `.pxd` files alongside augmented `.py` | Not needed -- pragma header is sufficient |
| 7 | `@cython.nogil` with Python objects | No Python objects in nogil context |
| 8 | `sys.modules` collision between variants | Namespace keys per variant |

## Build Integration

In `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.18.0", "hatch-cython>=0.6.0", "numpy>=1.20.0"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel.hooks.cython]
dependencies = ["hatch-cython>=0.6.0", "numpy>=1.20.0"]

[tool.hatch.build.targets.wheel.hooks.cython.options]
compile_py = false
include_numpy = true
directives = { boundscheck = false, nonecheck = false, language_level = 3, binding = true }
define_macros = [["NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"]]
```

The `.pyx` symlinks (created by the pre-commit hook) are what hatch-cython discovers and compiles.

## Migration Priority

1. **Performance metrics** -- tight numerical loops (drawdown, Sharpe, profit-factor)
2. **Candles data types** -- data-intensive processing (CandleData, utils)
3. **ConnectorBase / ExchangeBase** -- core infrastructure, biggest impact
4. **strategy_base / order_tracker** -- strategy execution hot path

## References

- [Cython Pure Python Mode](https://cython.readthedocs.io/en/latest/src/tutorial/pure.html)
- `sub-packages/cython-framework/` -- hb-cython-framework (test framework + build config)
- `dev/cython_candles` branch -- reference implementation (Coinbase candles triple-layout)
- `.claude/skills/cython-transition/` -- Claude Code skill for guided conversion
