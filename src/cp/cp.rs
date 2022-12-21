
//! Transductive deterministic or smooth Conformal Predictors.
use pcg_rand::Pcg32;
use rand::{Rng, SeedableRng};
use rusty_machine::learning::LearningResult;
use ndarray::prelude::*;
use std::f64::NAN;
use std::marker::PhantomData;

use cp::ConfidencePredictor;
use ncm::NonconformityScorer;


/// A Conformal Predictor, for some nonconformity scorer N and
/// matrix element type T.
///
/// CP can either be deterministic, where `smooth=false` and `rng=None`,
/// or smooth, where `smooth=true` and rng is some pseudo random number
/// generator (PRNG).
/// Let `Y` be a list of values ("prediction region") predicted by a CP
/// for a test input vector `x` with true label `y`.
/// If CP is constructed as deterministic, then:
/// $Pr(y \notin Y) \leq \varepsilon$, where $\varepsilon$ is the specified
/// significance level `epsilon`;
/// if CP is smooth, then:
/// $Pr(y \notin Y) = \varepsilon$.
pub struct CP<T: Sync, N: NonconformityScorer<T>> {
    ncm: N,
    epsilon: Option<f64>,
    smooth: bool,
    rng: Option<Pcg32>,
    n_labels: usize,
    // If calibrated is Some, this is an ICP, otherwise a TCP.
    calibrated: Option<bool>,
    // TODO: remove the following
    marker: PhantomData<T>,
}

impl<T: Sync, N: NonconformityScorer<T>> CP<T, N> {
    /// Constructs a new deterministic Transductive Conformal Predictor
    /// `CP<T,N>` from a nonconformity score NonconformityScorer.
    ///
    /// # Arguments
    ///
    /// * `ncm` - An object implementing NonconformityScorer.
    /// * `n_labels` - The number of labels.
    /// * `epsilon` - Either Some() significance level in [0,1] or None.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_world::cp::*;
    /// use random_world::ncm::*;
    ///
    /// let ncm = KNN::new(2);
    /// let n_labels = 2;
    /// let epsilon = 0.1;
    /// let mut cp = CP::new(ncm, n_labels, Some(epsilon));
    /// ```
    pub fn new(ncm: N, n_labels: usize, epsilon: Option<f64>) -> CP<T, N> {
        assert!(n_labels > 0);

        if let Some(e) = epsilon {
            assert!(e >= 0. && e <= 1.);
        }

        CP {
            ncm: ncm,
            epsilon: epsilon,
            smooth: false,
            n_labels: n_labels,
            rng: None,
            calibrated: None,
            marker: PhantomData,
        }
    }

    /// Constructs a new smooth Transductive Conformal Predictor
    /// `CP<T,N>` from a nonconformity score NonconformityScorer.
    ///
    /// # Arguments
    ///
    /// * `ncm` - An object implementing NonconformityScorer.
    /// * `n_labels` - The number of labels.
    /// * `epsilon` - Either Some() significance level in [0,1] or None.
    /// * `seed` - Optionally, a slice of 2 elements is provided as seed
    ///            to the random number generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_world::cp::*;
    /// use random_world::ncm::*;
    ///
    /// let ncm = KNN::new(2);
    /// let n_labels = 2;
    /// let epsilon = 0.1;
    /// let seed = [0, 0];
    /// let mut cp = CP::new_smooth(ncm, n_labels, Some(epsilon), Some(seed));
    /// ```
    pub fn new_smooth(ncm: N, n_labels: usize, epsilon: Option<f64>,
                      seed: Option<[u64; 2]>) -> CP<T, N> {

        if let Some(e) = epsilon {
            assert!(e >= 0. && e <= 1.);
        }

        CP {
            ncm: ncm,
            epsilon: epsilon,
            smooth: true,
            n_labels: n_labels,
            rng: match seed {
                Some(seed) => Some(Pcg32::from_seed(seed)),
                None => Some(Pcg32::new_unseeded())
            },
            calibrated: None,
            marker: PhantomData,
        }
    }

    /// Constructs a new deterministic Inductive Conformal Predictor
    /// `CP<T,N>` from a nonconformity score NonconformityScorer.
    ///
    /// # Arguments
    ///
    /// * `ncm` - An object implementing NonconformityScorer.
    /// * `n_labels` - The number of labels.
    /// * `epsilon` - Either Some() significance level in [0,1] or None.
    ///