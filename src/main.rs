mod mnist_loader;
mod network;
use mnist_loader::*;
use ndarray::*;
use network::*;

fn main() {
    let mut training_data = load_data("mnist/train").unwrap();
    // // We split the 60_000 training images into a training set and a validation set.
    let training_set = &mut training_data[0..50_000];

    let mut network = Network::new([2, 3, 1].to_vec());

    // let input = Array2::<f64>::from_shape_vec((2, 1), [5.0, 7.0].to_vec()).unwrap();
    network.sgd(training_set, 1, 10, 0.1);
    // network.feedforward(input);
}
