extern crate ndarray;
extern crate itertools;

extern crate random_world;

#[cfg(test)]
mod tests {
    use ndarray::*;
    use random_world::cp::*;
    use random_world::ncm::*;
    use random_world::utils::*;
    use itertools::multizip;
    
    #[test]
    fn cp() {
        let k = 2;
        let ncm = KNN::new(k);
        let n_labels = 2;
        let mut cp = CP::new(ncm, n_labels, Some(0.1));

        let train_inputs = array![[0., 0.],
                                  [1., 0.],
                                  [0., 1.],
                                  [1., 1.],
                                  [2., 2.],
                                  [1., 2.]];
        let train_targets = array![0, 0, 0, 1, 1, 1];
        let test_inputs = array![[2., 1.],
                                 [2., 2.]];
        let expected_pvalues = array![[0.25, 1.],
      