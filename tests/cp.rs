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
        let mut cp = CP::new