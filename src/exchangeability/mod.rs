//! Defines Exchangeability Martingales.
//!
//! Exchangeability martingales are tools for testing the
//! exchangeability (and i.i.d.) assumption.
//!
//! # Examples
//!
//! ```
//! extern crate rand;
//! extern crate statrs;
//! #[macro_use(array)]
//! extern crate ndarray;
//! extern crate random_world;
//! extern crate pcg_rand;
//!
//! # fn main() {
//! use pcg_rand::Pcg32;
//! use ndarray::prelude::*;
//! use rand::{Rng, SeedableRng};
//! use statrs::distribution::{Normal, Distribution};
//!
//! use random_world::cp::*;
//! use random_world::ncm::*;
//! use random_world::exchangeability::*;
//!
//! // Generate sequence. Trend change after 100 examples.
//! let seed = [0, 0];
//! let mut rng = Pcg32::from_seed(seed);
//! let n = Normal