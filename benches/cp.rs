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

fn generate_data(n: usize, d: usize, n_labels: usize, seed: [u64; 2])
        -> (Array2<f64>, Array1<usize>) {
    let mut rng = Pcg32::from_seed(seed);

    let inputs = Array::from_iter(rng.gen_iter::<f64>()
                                     .take(n*d)).into_shape((n, d))
                                                .unwrap();
    let targets = Array::from_iter((0..n).into_iter()
                                         .map(|_| rng.gen_range::<usize>(0, n_labels)));
    (inputs, targets)
}
    
#[bench