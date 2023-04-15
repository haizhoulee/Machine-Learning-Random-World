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
/