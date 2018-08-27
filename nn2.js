function sigmoid(x)
{
    return 1/(1+Math.exp(-x));
}
function dsigmoid(y)
{
    return y * (1-y);
    //return sigmoid(x) * (1 - sigmoid(x));
}
class NeuralNetwork
{
    constructor(input_nodes, hidden_nodes, output_nodes)
    {
        this.input_nodes=input_nodes;
        this.hidden_nodes=hidden_nodes;
        this.output_nodes=output_nodes;
        this.weights_ih=new Matrix(this.hidden_nodes,this.input_nodes);
        this.weights_ho=new Matrix(this.output_nodes,this.hidden_nodes);
        this.weights_ih.randomize();
        this.weights_ho.randomize();

        this.bias_h=new Matrix(this.hidden_nodes,1);
        this.bias_o=new Matrix(this.output_nodes,1);
        this.bias_h.randomize();
        this.bias_o.randomize();
        this.learning_rate=0.1;

    }
    feedforward(input_array)
    {
  
        let input=Matrix.fromArray(input_array);
        let hidden=Matrix.multiply(this.weights_ih,input);
        hidden.add(this.bias_h);
        hidden.map(sigmoid);
        let output=Matrix.multiply(this.weights_ho,hidden);
        output.add(this.bias_o);
        output.map(sigmoid);
        return output.toArray();
    }
    train(input_array,target_array)
    {
        let input=Matrix.fromArray(input_array);
        let hidden=Matrix.multiply(this.weights_ih,input);
        hidden.add(this.bias_h);
        hidden.map(sigmoid);
        let outputs=Matrix.multiply(this.weights_ho,hidden);
        outputs.add(this.bias_o);
        outputs.map(sigmoid);


        let targets=Matrix.fromArray(target_array);
        let output_errors=Matrix.subtract(targets,outputs);


        let gradients=Matrix.map(outputs,dsigmoid);
        gradients.multiply(output_errors);
        gradients.multiply(this.learning_rate);

        let hidden_T=Matrix.transpose(hidden);
        let weights_ho_deltas=Matrix.multiply(gradients,hidden_T);
        this.weights_ho.add(weights_ho_deltas);
        this.bias_o.add(gradients);

        let who_t=Matrix.transpose(this.weights_ho);
        let hidden_errors=Matrix.multiply(who_t,output_errors);
        let hidden_gradient=Matrix.map(hidden,dsigmoid);
        hidden_gradient.multiply(hidden_errors);
        hidden_gradient.multiply(this.learning_rate);
        let inputs_T= Matrix.transpose(input);

        let weights_ih_deltas=Matrix.multiply(hidden_gradient,inputs_T);
        this.weights_ih.add(weights_ih_deltas);
        this.bias_h.add(hidden_gradient);


    }
}