//! k-NN nonconformity measure.
use std::f64;
use std::cmp::min;
use lazysort::SortedPartial;
use ndarray::prelude::*;
use rusty_machine::learning::LearningResult;

use ncm::NonconformityScorer;


/// Returns the Euclidean distance between two vectors of f64 values.
fn euclidean_distance(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> f64 {
    v1.iter()
      .zip(v2.iter())
      .map(|(x,y)| (x - y).powi(2))
      .sum::<f64>()
      .sqrt()
}

/// Splits inputs according to their labels.
///
/// Returns as output a `train_inputs: Vec<Array2<T>>`, such that for each
/// unique label `y`, `train_inputs[y]` contains a matrix with the inputs with
/// label `y`.
fn split_inputs<T>(inputs: &ArrayView2<T>, targets: &ArrayView1<usize>,
         n_labels: usize) -> Vec<Array2<T>> where T: Clone + Sync + Copy {
    // Split examples w.r.t. their labels. For each unique label y,
    // train_inputs[y] will contain a matrix with the inputs with
    // label y.
    // We first put them into a vector, and then will convert them
    // into array. This should guarantee memory contiguity.
    // XXX: there may exist a better (faster) way.
    let mut train_inputs_vec = vec![vec![]; n_labels];

    for (x, y) in inputs.outer_iter().zip(targets) {
        // Implicitly asserts that 0 <= y < self.n_labels.
        train_inputs_vec[*y].extend(x.iter());
    }

    let d = inputs.cols();

    // Convert into arrays.
    let mut train_inputs = vec![];
    for inputs_y in train_inputs_vec {
        let n = inputs_y.len() / d;
        train_inputs.push(Array::from_shape_vec((n, d), inputs_y)
                                .expect("Unexpected error in reshaping"));
    }

    train_inputs
}

/// A k-NN nonconformity measure.
///
/// The score is defined for some distance metric and number of
/// neighbors.
pub struct KNN<T: Sync> {
    k: usize,
    distance: fn(&ArrayView1<T>, &ArrayView1<T>) -> f64,
    n_labels: Option<usize>,
    // Training inputs are stored in a train_inputs, indexed
    // by a label y, where train_inputs[y] contains all training
    // inputs with label y.
    train_inputs: Option<Vec<Array2<T>>>,
    // Calibration inputs are optional. If set, then the
    // NCM is trained on train_inputs, and the scores are
    // computed on calibration_inputs.
    calibration_inputs: Option<Vec<Array2<T>>>,
}

impl KNN<f64> {
    /// Constructs a k-NN nonconformity measure.
    ///
    /// # Arguments
    ///
    /// `k` - Number of nearest neighbors.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_world::ncm::*;
    ///
    /// let k = 2;
    /// let ncm = KNN::new(k);
    /// ```
    pub fn new(k: usize) -> KNN<f64> {
        KNN {
            k: k,
            distance: euclidean_distance,
            train_inputs: None,
            calibration_inputs: None,
            n_labels: None,
        }
    }
}

impl<T: Sync> NonconformityScorer<T> for KNN<T>
        where T: Clone + Sync + Copy {
    /// Trains a k-NN nonconformity scorer.
    ///
    /// Note: `train()` should be only called once. To update the training
    /// data of the `NonconformityScorer` use `update()`.
    ///
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix (Array2<T>) with values of type T of training
    ///              vectors.
    /// * `targets` - Vector (Array1<T>) of labels corresponding to the
    ///               t