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
    /// * `epsilon` - Parameter of the Power martingale.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_world::exchangeability::*;
    ///
    /// let epsilon = 0.95;
    /// let mut m = Martingale::new_power(epsilon);
    /// ```
    pub fn new_power(epsilon: f64) -> Martingale {
        assert!(epsilon >= 0.0 && epsilon <= 1.0);

        Martingale {
            update_function: Box::new(move |pvalue, _| {
                                        epsilon*pvalue.powf(epsilon-1.0)
                                    }),
            ..Default::default()
        }
    }

    /// Creates a new Simple Mixture martingale.
    ///
    /// NOTE: this is currently not implemented.
    pub fn new_simple_mixture() -> Martingale {
        unimplemented!();
    }

    /// Creates a new Plug-in martingale.
    ///
    /// To estimate the density it uses KDE with a gaussian kernel.
    /// If bandwidth is not specified, it uses Silverman's rule of thumb
    /// to determine its value.
    ///
    /// # Arguments
    ///
    /// * `bandwidth` - Bandwidth for the gaussian kernel in KDE. Can be None.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_world::exchangeability::*;
    ///
    /// let bandwidth = 0.2;
    /// let mut m = Martingale::new_plugin(Some(bandwidth));
    /// ```
    pub fn new_plugin(bandwidth: Option<f64>) -> Martingale {
        Martingale {
            pvalues: Some(vec![]),
            update_function: Box::new(move |pvalue, pvalues| {
                                       plugin_update(pvalue,
                                           &pvalues.as_ref()
                                                   .expect("Plug-in martingale badly initialized"),
                                           bandwidth)
                                     }),
            ..Default::default()
        }
    }

    /// Creates a new martingale from a custom update function.
    ///
    /// # Arguments
    ///
    /// * `u