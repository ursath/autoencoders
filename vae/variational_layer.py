import numpy as np
from .variation_multilayer_perceptron import MultiLayerPerceptron,LatentSpaceMultiLayerPerceptron

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
    
    def backward(self, delta_next, beta=1.0):
        derivative = self.prime_activation_function(self.h_j_values, beta) 
        self.last_delta = delta_next * derivative
        return self.weights_matrix[:, 1:].T @ self.last_delta

    def update_weights(self, learning_rate, optimizer=None, input_override=None, m_k=None, v_k=None, epoch=None, alpha=0.0, beta1=0.9, beta2=0.999, eps=1e-6, prev_dw=None):
        x = self.last_input if input_override is None else input_override
        for j in range(len(self.last_delta)):
            for i in range(len(x) - 1):
                grad = self.last_delta[j] * x[i]
                if optimizer is None:
                    delta_w = learning_rate * grad
                elif optimizer.__name__ == 'adam_optimizer_with_delta':
                    delta_w, m_k[j][i], v_k[j][i] = optimizer(self.last_delta[j], x[i], alpha, beta1, beta2, eps, m_k[j][i], v_k[j][i], epoch)
                elif optimizer.__name__ == 'momentum_gradient_descent_optimizer_with_delta':
                    delta_w = optimizer(learning_rate, self.last_delta[j], x[i], prev_dw[j][i], alpha)
                    prev_dw[j][i] = delta_w
                else:
                    delta_w = optimizer(learning_rate, self.last_delta[j], x[i], alpha)

                self.weights_matrix[j][i + 1] += delta_w
    
class LatentSpaceLayer():
    def __init__(self, num_neurons:int, activation_function, prime_activation_function=None, seed: int = 43):
        self.activation_function = activation_function
        self.prime_activation_function = prime_activation_function
        self.a_j_values = None
        self.neurons: list[LatentSpaceMultiLayerPerceptron] = []
        for i in range(num_neurons):
            perceptron = LatentSpaceMultiLayerPerceptron(2, activation_function)
            self.neurons.append(perceptron)

    def forward(self,medias,variances, beta=1.0):
        activated_outputs = np.array([])
        for i, neuron in enumerate(self.neurons):
            a_j, h_j = neuron.predict(np.array([medias[i], variances[i]]), np.array([1, 1]), beta)
            activated_outputs = np.append(activated_outputs, a_j)
        self.a_j_values = activated_outputs
        return activated_outputs

