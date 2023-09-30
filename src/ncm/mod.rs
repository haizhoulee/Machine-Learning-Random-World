
//! Module defining nonconformity measures.
//!
//! A `NonconformityScorer<T>` implements a nonconformity measure `score()`,
//! which determines how "strange" a new input vector looks like with
//! respect to previously observed ones.
pub mod knn;

use ndarray::prelude::*;
use rusty_machine::learning::LearningResult;

pub use self::knn::KNN;

/// A NonconformityScorer can be used to associate a
/// nonconformity score to a new example.
///
/// This trait is parametrized over `T`, the element type.
pub trait NonconformityScorer<T: Sync> {
    /// Trains a `NonconformityScorer`.
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
    ///               training vectors.
    fn train(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>,
             n_labels: usize) -> LearningResult<()>;
    /// Calibrates a `NonconformityScorer` for an ICP.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix (Array2<T>) with values of type T of training
    ///              vectors.
    /// * `targets` - Vector (Array1<T>) of labels corresponding to the
    ///               training vectors.
    fn calibrate(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>)
             -> LearningResult<()>;
    /// Updates a `NonconformityScorer` with more training data.
    ///
    /// After calling `train()` once, `update()` allows to add
    /// inputs to the scorer's training data, which will be used
    /// for future predictions.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix (Array2<T>) with values of type T of training
    ///              vectors.
    /// * `targets` - Vector (Array1<T>) of labels corresponding to the
    ///               training vectors.
    fn update(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>) -> LearningResult<()>;