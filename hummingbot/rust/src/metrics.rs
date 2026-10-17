use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Calculate max drawdown and max drawdown percentage from cumulative P&L series.
///
/// Returns (max_drawdown_abs, max_drawdown_pct)
#[pyfunction]
fn calculate_max_drawdown(pnl_series: PyReadonlyArray1<f64>) -> PyResult<(f64, f64)> {
    let data: ArrayView1<f64> = pnl_series.as_array();
    let n = data.len();
    if n == 0 {
        return Ok((0.0, 0.0));
    }

    let mut peak = data[0];
    let mut max_dd = 0.0_f64;
    let mut max_dd_pct = 0.0_f64;

    for i in 1..n {
        if data[i] > peak {
            peak = data[i];
        }
        let dd = peak - data[i];
        if dd > max_dd {
            max_dd = dd;
            if peak != 0.0 {
                max_dd_pct = dd / peak;
            }
        }
    }

    Ok((max_dd, max_dd_pct))
}

/// Calculate annualized Sharpe ratio from returns series.
///
/// Uses Welford's online algorithm for numerical stability.
#[pyfunction]
#[pyo3(signature = (returns, periods_per_year=252))]
fn calculate_sharpe_ratio(returns: PyReadonlyArray1<f64>, periods_per_year: usize) -> PyResult<f64> {
    let data: ArrayView1<f64> = returns.as_array();
    let n = data.len();
    if n < 2 {
        return Ok(0.0);
    }

    // Welford's online algorithm for mean and variance
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;

    for i in 0..n {
        let delta = data[i] - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = data[i] - mean;
        m2 += delta * delta2;
    }

    let variance = m2 / (n - 1) as f64;
    let std = variance.sqrt();

    if std == 0.0 {
        return Ok(0.0);
    }

    Ok((mean / std) * (periods_per_year as f64).sqrt())
}

/// Calculate profit factor (sum of wins / abs sum of losses).
#[pyfunction]
fn calculate_profit_factor(pnl_values: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let data: ArrayView1<f64> = pnl_values.as_array();
    let mut gross_profit = 0.0_f64;
    let mut gross_loss = 0.0_f64;

    for &val in data.iter() {
        if val > 0.0 {
            gross_profit += val;
        } else {
            gross_loss += val.abs();
        }
    }

    if gross_loss == 0.0 {
        return Ok(f64::INFINITY);
    }

    Ok(gross_profit / gross_loss)
}

/// Calculate all performance metrics in a single pass.
///
/// Returns (max_drawdown, max_drawdown_pct, sharpe_ratio, profit_factor)
#[pyfunction]
#[pyo3(signature = (pnl_series, periods_per_year=252))]
fn calculate_all_metrics(
    pnl_series: PyReadonlyArray1<f64>,
    periods_per_year: usize,
) -> PyResult<(f64, f64, f64, f64)> {
    let data: ArrayView1<f64> = pnl_series.as_array();
    let n = data.len();
    if n == 0 {
        return Ok((0.0, 0.0, 0.0, 0.0));
    }

    // Drawdown tracking
    let mut peak = data[0];
    let mut max_dd = 0.0_f64;
    let mut max_dd_pct = 0.0_f64;

    // Welford's for Sharpe
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;

    // Profit factor accumulators
    let mut gross_profit = 0.0_f64;
    let mut gross_loss = 0.0_f64;

    for i in 0..n {
        let val = data[i];

        // Drawdown
        if i > 0 {
            if val > peak {
                peak = val;
            }
            let dd = peak - val;
            if dd > max_dd {
                max_dd = dd;
                if peak != 0.0 {
                    max_dd_pct = dd / peak;
                }
            }
        }

        // Welford's
        let delta = val - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = val - mean;
        m2 += delta * delta2;

        // Profit factor
        if val > 0.0 {
            gross_profit += val;
        } else {
            gross_loss += val.abs();
        }
    }

    let sharpe = if n >= 2 {
        let variance = m2 / (n - 1) as f64;
        let std = variance.sqrt();
        if std == 0.0 {
            0.0
        } else {
            (mean / std) * (periods_per_year as f64).sqrt()
        }
    } else {
        0.0
    };

    let pf = if gross_loss == 0.0 {
        f64::INFINITY
    } else {
        gross_profit / gross_loss
    };

    Ok((max_dd, max_dd_pct, sharpe, pf))
}

/// Register metrics functions on the given PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let metrics_module = PyModule::new(m.py(), "metrics")?;
    metrics_module.add_function(wrap_pyfunction!(calculate_max_drawdown, &metrics_module)?)?;
    metrics_module.add_function(wrap_pyfunction!(calculate_sharpe_ratio, &metrics_module)?)?;
    metrics_module.add_function(wrap_pyfunction!(calculate_profit_factor, &metrics_module)?)?;
    metrics_module.add_function(wrap_pyfunction!(calculate_all_metrics, &metrics_module)?)?;
    m.add_submodule(&metrics_module)?;
    Ok(())
}
