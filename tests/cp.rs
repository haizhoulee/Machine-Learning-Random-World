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
                                      [0.25, 1.]];

        let epsilon_1 = 0.3;
        let epsilon_2 = 0.2;
        let expected_preds_1 = array![[false, true],
                                      [false, true]];
        let expected_preds_2 = array![[true, true],
                                      [true, true]];

        cp.train(&train_inputs.view(), &train_targets.view()).unwrap();
        let pvalues = cp.predict_confidence(&test_inputs.view()).unwrap();
        println!("Expected p-values: {:?}", expected_pval