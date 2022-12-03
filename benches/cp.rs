#![feature(test)]

extern crate test;
extern crate rand;
extern crate ndarray;
extern crate pcg_rand;
extern crate random_world;

use test::{Bencher, black_box};
use pcg_rand::Pcg32;
use rand::{Rng, SeedableRng};
use ndarray::prelude::*;

use random_world::cp::*;
use random_world::ncm::*;

fn gener