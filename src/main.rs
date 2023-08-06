#[derive(Debug)]
#[derive(Clone)]
#[derive(Copy)]
struct Variable {
    value: f32,
    derivative: f32
}

#[derive(Debug)]
#[derive(Clone)]
struct Neuron {
    bias: Variable,
    weights: Vec<Variable>,
    derivative: f32
}

#[derive(Debug)]
#[derive(Clone)]
struct Layer {
    neurons :Vec<Neuron>
}

#[derive(Clone)]
struct MLP{
    layers : Vec<Layer>
}

impl Variable{
    fn init(init_value: f32) -> Self {
        Self {
            value: init_value,
            derivative: 1.
        }
    }
    fn add(&mut self, other : &mut Variable) -> Variable{
        let new_value = Variable::init(self.value + other.value);
        self.derivative += 1.0;
        other.derivative += 1.;
        return new_value;
    }

    fn multiply(&mut self, other : &mut Variable ) -> Variable {
        let new_value = Variable::init(self.value * other.value);
        self.derivative += other.value;
        other.derivative += self.value;
        return new_value;
    }

    fn grad_descent(&mut self, lr : f32) {
        self.value = (-self.derivative * lr) + self.value;
        self.derivative = 1.0;
    }

    fn relu(&mut self) -> Variable{
        let new_value;
        if self.value > 0. {
            self.derivative += 1.;
            new_value = self.value;
        }
        else{
            self.derivative = 0.;
            new_value = 0.;
        }
        Variable::init(new_value)
    }

    fn exp(&mut self) -> Variable {
        let exp_value = self.value.exp();
        self.derivative += exp_value;
        Variable::init(exp_value)
    }

    fn div(&mut self, other : &mut Variable ) -> Variable {
        let new_value = self.value / other.value;
        self.derivative += 1. / other.value;
        other.derivative += -self.value / other.value.powi(2);
        Variable::init(new_value)
    }
}

impl Neuron{
    fn init(size: usize) -> Self{
        Self {
            weights: vec!{Variable::init(1000.); size},
            bias: Variable::init(1.),
            derivative: 1.0
        }
    }
    fn activation(&mut self, inputs: &mut Vec<Variable>) -> Variable {
        let mut output = Variable::init(0.);
        for index in 0..inputs.len()  {
            output = output.add(&mut inputs[index].multiply(&mut self.weights[index]));
        }
        output = output.add(&mut self.bias).relu();
        self.derivative *= output.derivative;
        return output;
    }
}

impl Layer {
    fn init(size : usize, prev_size : usize) -> Self {
        return Self { neurons: vec!{Neuron::init(prev_size); size}}
    }

    fn get_size(&self) -> usize{
        self.neurons.len()
    }

    fn activation(&mut self, input: &mut Vec<Variable>) -> Vec<Variable>{
        let mut outputs = Vec::new();
        for index in 0..self.neurons.len() {
            outputs.push( self.neurons[index].activation(input));
        }
        outputs
    }
}

impl MLP {
    fn init(shape: Vec<usize>, input : usize) -> Self {
        let mut layers = Vec::new();
        layers.push(Layer::init(shape[0], input));
        for index in 1..shape.len() {
            layers.push(Layer::init(shape[index], layers[layers.len()-1].get_size()))
        }
        return Self { layers : layers};
    }

    fn forward(&mut self, input : &mut Vec<Variable>) -> Vec<Variable>{
        let mut carrier: Vec<Variable>;
        carrier = self.layers[0].activation(input);
        for index in 1..self.layers.len(){
            carrier = self.layers[index].activation(&mut carrier);
        }
        carrier
    }

    fn back_prop(&mut self, output : Vec<Variable>, lr : f32) -> MLP {
        for i in (0..self.layers.len()).rev(){
            for j in 0..self.layers[i].neurons.len(){
                if i == self.layers.len()-1 {
                    self.layers[i].neurons[j].derivative *= output[j].derivative
                }
                else {
                    for prev_neuron in self.layers[i+1].neurons.clone(){
                        self.layers[i].neurons[j].derivative *= prev_neuron.weights[j].derivative;
                    }
                }
                for k in 0..self.layers[i].neurons[j].weights.len(){
                    self.layers[i].neurons[j].weights[k].derivative *= self.layers[i].neurons[j].derivative;
                    self.layers[i].neurons[j].weights[k].grad_descent(lr);
                }
                self.layers[i].neurons[j].bias.derivative *= self.layers[i].neurons[j].derivative;
                self.layers[i].neurons[j].bias.grad_descent(lr);
            }
        }
        return self.clone();
    }
}

// fn quad_loss(input : &mut Vec<Variable>, objective : &mut Vec<Variable>) -> Variable{

// }

fn main() {
    let mut network = MLP::init(vec!{1,2,1,1}, 2);
    let mut inputs = vec!{Variable::init(10.); 2};
    for _ in 0..1{
        let mut output = network.forward(&mut inputs);
        output[0].exp();
        output[0].add(&mut Variable::init(-10.0).exp());
        network = network.back_prop(output, 0.0001);
    }
    println!("{:?}", network.forward(&mut inputs));
}
