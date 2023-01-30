//! Martingales for exchangeability testing.
use std::f64;
use quadrature::integrate;
use statrs::statistics::Variance;
/// Exchangeability Martingale.
///
/// A generic exchangeability martingale, as described for example
/// in [1,2].
///
/// [1] "Testing Exchangeability On-Line" (Vovk et al., 2003).
/// [2] "Plug-in martingales for testing exchangeability on-line"
///     (Fedorova et al., 2012).
pub struct Martingale {
    /// Current value of the martingale.
    current: f64,
    /// Threshold to determine if the martingale is "large".
    pub threshold: f64,
    /// Some methods need to record previous p-values.
    pvalues: Option<Vec<f64>>,
    /// The martingale M is updated given a new p-value p
    /// as:
    ///     M *= update_function(p, pvalues)
    /// where pvalues are optionally recorded previous p-values.
    update_function: Box<Fn(f64, &Option<Vec<f64>>) -> f64>,
}

impl Default for Martingale {
    /// Default values for `Martingale`.
    ///
    /// NOTE: constructor methods (e.g., `new_power()`, `new_plugin()`)
    /// should be preferred to using defaults; default for
    /// `update_function`, for  example, is a meaningless placeholder
    /// function.
    /// If one wants to instantiate a `Martingale` with a custom
    /// `update_function`, they are recommended to use the
    /// `Martingale::from_function()` constructor.
    fn default() -> Martingale {
        Martingale {
            current: 1.0,
            threshold: 100.0,
            pvalues: None,
            // Placeholder update_function.
            update_function: Box::new(|_, _| { f64::NAN }),
        }
    }
}


impl Martingale {
    /// Creates a new Power martingale.
    ///
    /// # Arguments
    ///
    /// * `ep