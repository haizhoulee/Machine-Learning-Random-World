
#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate serde_derive;
extern crate docopt;
extern crate random_world;
extern crate itertools;

use random_world::cp::*;
use random_world::ncm::*;
use random_world::utils::{load_data, store_predictions};
use itertools::Itertools;
use docopt::Docopt;
use ndarray::*;

const USAGE: &'static str = "
Predict data using Conformal Prediction.

If no <testing-file> is specified, on-line mode is assumed.

Usage: cp-predict knn [--knn=<k>] [options] [--] <output-file> <training-file> [<testing-file>]
       cp-predict kde [--kernel<kernel>] [--bandwidth=<bw>] [options] [--] <output-file> <training-file> [<testing-file>]
       cp-predict (--help | --version)

Options:
    -e, --epsilon=<epsilon>     Significance level. If specified, the output are