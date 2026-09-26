use pyo3::prelude::*;

mod metrics;
mod order_expiration_entry;

#[pymodule]
fn _hb_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    metrics::register(m)?;
    order_expiration_entry::register(m)?;
    Ok(())
}
