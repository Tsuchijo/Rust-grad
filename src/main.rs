extern crate ndarray;
use ndarray::{Array, Array2};

#[derive(Debug)]
#[derive(Clone)]
enum Activation{
    Relu
}

#[derive(Debug)]
#[derive(Clone)]
struct Layer {
    weights : ndarray::Array2<f64>,
    bias : ndarray::Array1<f64>,
    delta : ndarray::Array1<f64>,
    input : ndarray::Array1<f64>, 
    hidden : ndarray::Array1<f64>,
    activation : Activation
}

struct Loss { 
    derivative : ndarray::Array1<f64>,
    loss : f64
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

    fn compute_delta_final(&mut self, loss : &Loss){
        self.delta = self.hidden.mapv(Layer::relu_d) * &loss.derivative;
    }

    fn backprop(&mut self, lr : &f64){
        let mut update_matrix: ndarray::Array2<f64> = Array::zeros((self.delta.len(), self.input.len()));
        for (i, delta) in self.delta.iter().enumerate() {
            for (j, input) in self.input.iter().enumerate() {
                update_matrix[[i, j]] = delta * input;
            }
        }
        self.weights += - (update_matrix);
        //self.bias = self.bias - (self.delta);
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
  fn init(layers : Vec<Layer>) -> Self {
    NeuralNetwork {
        layers : layers
    }
  }
  
  fn forward (&mut self, input : &ndarray::Array1<f64>) -> ndarray::Array1<f64>{
    let mut output = input.clone();
    for layer in &mut self.layers {
        output = layer.forward(&output);
    }
    output
  }

  fn backprop (&mut self, loss: &Loss, lr : &f64){
    let mut prev_layer = self.layers[self.layers.len() - 1].clone();
    let size = self.layers.len();
    self.layers[size - 1].compute_delta_final(loss);
    self.layers[size - 1].backprop(lr);
    for i in (0..self.layers.len() - 1).rev(){
        self.layers[i].compute_delta(prev_layer.clone());
        self.layers[i].backprop(lr);
        prev_layer = self.layers[i].clone();
    }
  }
}

impl Loss {
    fn init() -> Self {
        Loss { 
            derivative: ndarray::Array1::zeros(10),
            loss : 0.0 
            }
    }
    
    fn quadraticnn(&mut self, output : &ndarray::Array1<f64>, target : &ndarray::Array1<f64>) -> f64{
        let mut loss = 0.0;
        for i in 0..output.len(){
            loss += (output[i] - target[i]).powi(2);
        }
        //  compute derivative of loss
        self.derivative = output - target;
        self.loss = loss;
        loss
    }
}
fn main () {
     let mut layers = Vec::new();
    layers.push(Layer::init(784, 16));
    layers.push(Layer::init(16, 16));
    layers.push(Layer::init(16, 10));
    let mut nn = NeuralNetwork::init(layers);
    let mut loss = Loss::init();
    let mut input = ndarray::Array1::zeros(784);
    let mut target = ndarray::Array1::zeros(10);
    let mut output = ndarray::Array1::zeros(10);
    let mut lr = 0.01;
    for _ in 0..1000{
        input = Array::ones(784);
        target = Array::ones(10);
        output = nn.forward(&input);
        let loss_value = loss.quadraticnn(&output, &target);
        nn.backprop(&loss, &lr);
    }
    println!("input : {:?}", input);
    println!("target : {:?}", target);
    println!("output : {:?}", output);
}