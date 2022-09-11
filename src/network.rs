use ndarray::{Array, Array2};
use rand::{seq::SliceRandom, thread_rng};

use crate::mnist_loader::MnistImage;

pub struct Network {
    num_layers: usize,
    sizes: Vec<usize>, // Contains the number of neurons in each respective layer.
    biases: Vec<Array2<f64>>, // Vector of column representing the biases for each layer.
    // Vector of matrices representing weights between two layers i and i + 1.
    // Dimension of each vector is sizes[i+1] rows by sizes[i] columns.
    // Entry at [i][j] represents the weight from the ith layer the jth layer.
    // This configuration allows us to calculate the activation for the current layer as
    // a' = sigmoid(w * a + b)
    // where w is the weight matrix from layer i to i + 1, b is the bias for the current layer, and
    // a is the activation of the previous layer.
    // The weight matrix must have the same number of cols as the size of the previous layer, and
    // must have the number of rows equal to the size of the current layer.
    weights: Vec<Array2<f64>>,
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        let num_layers = sizes.len();

        // Initialize the bias vectors.
        let mut biases: Vec<Array2<f64>> = Vec::with_capacity(num_layers - 1); // input layer doesn't have bias.
        for y in sizes[1..num_layers].into_iter() {
            biases.push(Array2::zeros((*y, 1)));
        }

        // Initialize the weight matrices.
        let mut weights: Vec<Array2<f64>> = Vec::with_capacity(num_layers - 1);
        for (x, y) in sizes[0..num_layers]
            .into_iter()
            .zip(sizes[1..num_layers].into_iter())
        {
            let matrix_shape = (*y, *x);
            weights.push(Array2::from_shape_fn(matrix_shape, |(i, j)| {
                rand::random::<f64>() * 20.0 - 10.0 // TODO: Configure how to initialize the weights.
            }));
        }

        Self {
            num_layers,
            sizes,
            biases,
            weights,
        }
    }

    /// Given an input to the network, returns the corresponding output.
    pub fn feedforward(&self, mut a: Array2<f64>) -> Array2<f64> {
        // Check that the input dimensions are correct.
        debug_assert!(
            a.shape()[0] == self.weights[0].shape()[1],
            "Dimension mismatch!"
        );

        // Apply a' = sigmoid
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            a = sigmoid(w.dot(&a) + b);
        }
        println!("{:?}", a);
        a
    }

    /// Train the neural network using mini-baatch stochastic gradient descent.
    pub fn sgd(
        &mut self,
        training_data: &mut [MnistImage],
        epochs: usize,
        mini_batch_size: usize,
        eta: f64,
    ) {
        let mut rng = thread_rng();
        let n = training_data.len();
        for j in 0..epochs {
            // Shuffle training data.
            training_data.shuffle(&mut rng);

            // Split training data into mini batches.
            // TODO: Modify so that n doesn't have to be a multiple of the mini_batch_size
            let mut mini_batches: Vec<&[MnistImage]> = Vec::with_capacity(n / mini_batch_size);
            for k in (0..n).step_by(mini_batch_size) {
                mini_batches.push(&training_data[k..k + mini_batch_size]);
            }

            // Train network.
            for mini_batch in mini_batches {
                // Apply single step of gradient descent.
                self.update_mini_batch(mini_batch, eta);
            }

            // Print progress.
            println!("Epoch {} complete", j);
        }
    }

    /// Update the network's weights and biases by applying gradient descent using backpropogation to a single mini batch.
    /// "mini-batch" is a slice of training data and "eta" is the learning rate.
    fn update_mini_batch(&mut self, mini_batch: &[MnistImage], eta: f64) {
        // Partial derivatives of cost function w/ respect to the biases.
        let mut nabla_b: Vec<Array2<f64>> = Vec::with_capacity(self.biases.len());
        for b in &self.biases {
            let shape = b.shape();
            nabla_b.push(Array2::<f64>::zeros((shape[0], shape[1])));
        }

        // Partial derivatives of cost function w/ respect to the weights.
        let mut nabla_w: Vec<Array2<f64>> = Vec::with_capacity(self.weights.len());
        for w in &self.weights {
            let shape = w.shape();
            nabla_w.push(Array2::zeros((shape[0], shape[1])));
        }

        // Compute the gradient of the cost function for this mini batch.
        for training_image in mini_batch {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(training_image);



        }
    }

    /// Returns a tuple "(nabla_b, nabla_w)" representing the gradient for the cost function C_x. 
    /// "nabla_b" and "nabla_w" are layer-by-layer lists of vectors, similar to "self.biases" and "self.weights".
    fn backprop(
        &mut self,
        training_image: &MnistImage,
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_b: Vec<Array2<f64>> = Vec::with_capacity(self.biases.len());
        for b in &self.biases {
            let shape = b.shape();
            nabla_b.push(Array2::<f64>::zeros((shape[0], shape[1])));
        }
        let mut nabla_w: Vec<Array2<f64>> = Vec::with_capacity(self.weights.len());
        for w in &self.weights {
            let shape = w.shape();
            nabla_w.push(Array2::zeros((shape[0], shape[1])));
        }

        // Feedforward.
        let activation = &training_image.image;
        let mut activations = vec![activation]; // list to store all the activations, layer by layer.
        let mut zs: Vec<Array2<f64>> = Vec::new(); // List to store the z vectors, layer by layer.

        (nabla_b, nabla_w)
    }
}

// TODO: Check if passing by reference changes performance.
pub fn sigmoid(z: Array2<f64>) -> Array2<f64> {
    z.map(|x| 1.0 / (1.0 + f64::exp(*x)))
}
