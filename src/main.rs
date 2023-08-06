use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use rand::{prelude::*, distributions::Standard};

enum Activation{
    Relu
}

struct Layer {
    weights : ndarray::Array2<f64>,
    bias : ndarray::Array1<f64>,
    delta : ndarray::Array1<f64>,
    input : ndarray::Array1<f64>, 
    hidden : ndarray::Array1<f64>,
    activation : Activation
}

struct Loss { 
    derivative : ndarray::Array1<f64>
}

struct NeuralNetwork{
    layers : Vec<Layer>
}


impl Layer {

    fn relu(x : f64) -> f64{
        x.max(0.0)
    }

    fn relu_d(x : f64) -> f64{
        if x > 0.0 {
            1.0
        }
        else {
            0.0
        }
    }

    fn forward(&mut self, input : &ndarray::Array1<f64>) -> ndarray::Array1<f64>{
        self.hidden = self.weights.dot(input) + &self.bias;
        self.input = input.clone();
        match self.activation{
            Activation::Relu => {
                self.hidden.mapv(Layer::relu)
            }
        }
    }

    fn compute_delta(&mut self, prev_layer : Layer){
        match self.activation{
            Activation::Relu => {
                self.delta = prev_layer.weights.t().dot(&prev_layer.delta) * self.hidden.mapv(Layer::relu_d)
            }
            _=>{

            }
        }
    }

    fn compute_delta_final(&mut self, loss : Loss){
        self.delta = self.hidden.mapv(Layer::relu_d) * loss.derivative;
    }

    fn backprop(&mut self, lr : &f64){
        self.weights = &self.weights - self.delta.dot(&self.input.t()) * lr;
        //self.bias = &self.bias - ((&self.delta) * lr);
    }

    // init a layer with random weights and bias
    fn init(input_size : usize, output_size : usize) -> Self {
        Layer {
            weights : Array::ones((output_size, input_size)),
            bias : Array::ones(output_size),
            delta : ndarray::Array1::zeros(output_size),
            input : ndarray::Array1::zeros(input_size),
            hidden : ndarray::Array1::zeros(output_size),
            activation : Activation::Relu
        }
    }

}

impl NeuralNetwork {
    fn forward(&mut self, input : &ndarray::Array1<f64>) -> ndarray::Array1<f64>{
        let mut carrier = input;
        for mut layer in self.layers {
            carrier = &layer.forward(carrier);
        }
        carrier.clone()
    }

    fn backward(&mut self, loss : Loss){
        // Compute delta for last layer
        self.layers.last_mut().unwrap().compute_delta_final(loss);
        // Compute delta for all other layers
        for i in (0..self.layers.len()-1).rev() {
            self.layers[i].compute_delta(self.layers[i+1]);
        }
        // Backpropagate
        for  mut layer in self.layers {
            layer.backprop(&0.1);
        }
    }
}

impl Loss {
    fn init() -> Self {
        Loss { derivative: ndarray::Array1::zeros(10) }
    }
    
    fn quadraticnn(&mut self, output : ndarray::Array1<f64>, target : ndarray::Array1<f64>) -> f64{
        let mut loss = 0.0;
        for i in 0..output.len(){
            loss += (output[i] - target[i]).powi(2);
        }
        //  compute derivative of loss
        self.derivative = output - target;
        loss
    }
}
fn main () {
    // Generate test dataset
    let mut rng = rand::thread_rng();
    let mut input = Array::ones(10);
    let mut target = Array::ones(10);
    // Initialize neural network
    let mut nn = NeuralNetwork { layers: vec![Layer::init(10, 10), Layer::init(10, 10)] };
    // Forward pass
    let output = nn.forward(&input);
    // Compute loss
    let mut loss = Loss::init();
    let loss_value = loss.quadraticnn(output, target);
    // Backward pass
    nn.backward(loss);
}