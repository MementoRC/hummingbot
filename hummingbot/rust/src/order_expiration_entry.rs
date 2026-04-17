use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyType};
use pyo3::Borrowed;

/// Rust implementation of hummingbot.core.data_type.OrderExpirationEntry.
///
/// Maps the C++ struct directly:
///   std::string tradingPair        → String
///   std::string orderId            → String
///   double      timestamp          → f64
///   double      expiration_timestamp → f64
///
/// Ordering (C++ operator<) is preserved as __lt__:
///   primary key:   expiration_timestamp ascending
///   secondary key: order_id ascending (lexicographic)
#[pyclass(module = "_hb_rust.order_expiration_entry", from_py_object)]
#[derive(Clone, Debug)]
pub struct OrderExpirationEntry {
    trading_pair: String,
    order_id: String,
    timestamp: f64,
    expiration_timestamp: f64,
}

#[pymethods]
impl OrderExpirationEntry {
    /// Create a new OrderExpirationEntry.
    ///
    /// Args:
    ///     trading_pair:  Trading pair identifier (e.g. "BTC-USDT")
    ///     order_id:      Client order ID string
    ///     timestamp:     Order creation timestamp (Unix seconds)
    ///     expiration_ts: Order expiration timestamp (Unix seconds)
    #[new]
    pub fn new(
        trading_pair: String,
        order_id: String,
        timestamp: f64,
        expiration_ts: f64,
    ) -> Self {
        OrderExpirationEntry {
            trading_pair,
            order_id,
            timestamp,
            expiration_timestamp: expiration_ts,
        }
    }

    /// Trading pair identifier.
    #[getter]
    pub fn trading_pair(&self) -> &str {
        &self.trading_pair
    }

    /// Client order ID.
    #[getter]
    pub fn order_id(&self) -> &str {
        &self.order_id
    }

    /// Order creation timestamp (Unix seconds).
    #[getter]
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }

    /// Order expiration timestamp (Unix seconds).
    #[getter]
    pub fn expiration_timestamp(&self) -> f64 {
        self.expiration_timestamp
    }

    /// String representation matching the original Cython __repr__.
    pub fn __repr__(&self) -> String {
        format!(
            "OrderExpirationEntry('{}', '{}', {}, '{}')",
            self.trading_pair, self.order_id, self.timestamp, self.expiration_timestamp
        )
    }

    /// Rich comparison: less-than.
    ///
    /// Matches C++ operator<:
    ///   - primary sort key:   expiration_timestamp (ascending)
    ///   - secondary sort key: order_id (lexicographic ascending)
    pub fn __lt__(&self, other: &OrderExpirationEntry) -> bool {
        if self.expiration_timestamp == other.expiration_timestamp {
            self.order_id < other.order_id
        } else {
            self.expiration_timestamp < other.expiration_timestamp
        }
    }

    /// Convert a list of OrderExpirationEntry objects to a pandas DataFrame.
    ///
    /// Columns: ["trading_pair", "order_id", "timestamp", "expiration_timestamp"]
    ///
    /// Note: the original Cython implementation referenced non-existent fields
    /// `expiration_entry.expiration` and `expiration_entry.expiration_time`.
    /// This implementation fixes that bug by using `expiration_timestamp` consistently.
    #[classmethod]
    pub fn to_pandas<'py>(
        _cls: &Bound<'py, PyType>,
        py: Python<'py>,
        order_expiration_entries: Vec<PyRef<'py, OrderExpirationEntry>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pd = py.import("pandas")?;

        let columns = PyList::new(
            py,
            ["trading_pair", "order_id", "timestamp", "expiration_timestamp"],
        )?;

        let data = PyList::empty(py);
        for entry in &order_expiration_entries {
            let row = PyList::new(
                py,
                [
                    entry.trading_pair.clone().into_pyobject(py)?.into_any(),
                    entry.order_id.clone().into_pyobject(py)?.into_any(),
                    entry.timestamp.into_pyobject(py)?.into_any(),
                    entry.expiration_timestamp.into_pyobject(py)?.into_any(),
                ],
            )?;
            data.append(row)?;
        }

        let kwargs = PyDict::new(py);
        kwargs.set_item("data", data)?;
        kwargs.set_item("columns", columns)?;
        pd.getattr("DataFrame")?.call((), Some(&kwargs))
    }
}

/// Register order_expiration_entry types on the given PyModule.
///
/// Called by the root _hb_rust module at integration time:
///   let sub = PyModule::new(m.py(), "order_expiration_entry")?;
///   order_expiration_entry::register(sub.as_borrowed())?;
///   m.add_submodule(&sub)?;
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let oee_module = PyModule::new(m.py(), "order_expiration_entry")?;
    oee_module.add_class::<OrderExpirationEntry>()?;
    m.add_submodule(&oee_module)?;
    Ok(())
}
