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
    /