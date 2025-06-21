import numpy as np
from .multilayer_perceptron import MultiLayerPerceptron

class Layer:
    # num inputs len(weight)
    # num_neurons = k
    # num_previous_layer_neurons = n
    def __init__(self, num_inputs:int ,num_previous_layer_neurons:int, num_neurons: int, activation_function, prime_activation_function=None, seed: int = 43):
        self.activation_function = activation_function  
        self.prime_activation_function = prime_activation_function
        # wi0 is for bias
        # now each weight has a shape of 0
        self.weights_matrix = np.random.rand(num_neurons, num_previous_layer_neurons + 1) * 0.1
        self.a_j_values = None
        self.h_j_values = None
        self.neurons: list[MultiLayerPerceptron] = []
        for i in range(num_neurons):
            perceptron = MultiLayerPerceptron(num_previous_layer_neurons, activation_function)
            self.neurons.append(perceptron)

    def forward(self, inputs, beta=1.0):
        activated_outputs = np.array([])
        h_values = np.array([])
        input_with_bias = np.insert(inputs, 0, 1)  # añado el bias como x0 = 1
        for i, neuron in enumerate(self.neurons):
            a_j, h_j = neuron.predict(input_with_bias, self.weights_matrix[i], beta)
            h_values = np.append(h_values, h_j)
            activated_outputs = np.append(activated_outputs, a_j)
        self.a_j_values = activated_outputs        
        self.h_j_values = h_values
        return activated_outputs
