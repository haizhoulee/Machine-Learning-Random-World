//! Martingales for exchangeability testing.
use std::f64;
use quadrature::integrate;
use statrs::statistics::Variance;
/// Exchangeability Martingale.
///
/// A generic exchangeability martingale, as described for example
/// in