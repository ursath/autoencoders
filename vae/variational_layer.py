import numpy as np
from .variation_multilayer_perceptron import MultiLayerPerceptron

class Layer:
    def __init__(self, num_inputs:int ,num_previous_layer_neurons:int, num_neurons: int, activation_function, prime_activation_function=None, seed: int = 43):
        self.activation_function = activation_function  
        self.prime_activation_function = prime_activation_function
        self.weights_matrix = np.random.rand(num_neurons, num_previous_layer_neurons + 1) * 0.1
        self.a_j_values = None
        self.h_j_values = None
        self.last_input = None
        self.last_delta = None
        self.neurons: list[MultiLayerPerceptron] = [
            MultiLayerPerceptron(num_previous_layer_neurons, activation_function)
            for _ in range(num_neurons)
        ]

    def forward(self, inputs, beta=1.0):
        input_with_bias = np.insert(inputs, 0, 1)  # Añadir bias
        self.last_input = input_with_bias
        activated_outputs = []
        h_values = []
        for i, neuron in enumerate(self.neurons):
            a_j, h_j = neuron.predict(input_with_bias, self.weights_matrix[i], beta)
            activated_outputs.append(a_j)
            h_values.append(h_j)
        self.a_j_values = np.array(activated_outputs)
        self.h_j_values = np.array(h_values)
        return self.a_j_values
    
    def backward(self, delta_next, beta=1.0):
        derivative = self.prime_activation_function(self.h_j_values, beta)
        self.last_delta = delta_next * derivative
        return self.weights_matrix[:, 1:].T @ self.last_delta

    def update_weights(self, learning_rate, optimizer=None, input_override=None, m_k=None, v_k=None, epoch=None, alpha=0.0, beta1=0.9, beta2=0.999, eps=1e-6, prev_dw=None):
        x = self.last_input if input_override is None else np.insert(input_override, 0, 1)
        for j in range(len(self.last_delta)):
            for i in range(len(x)):
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
                self.weights_matrix[j][i] += delta_w


class LatentSpaceLayer:
    def __init__(self):
        self.last_mu = None
        self.last_logvar = None
        self.last_z = None
        self.last_epsilon = None

    def forward(self, mu, logvar):
        self.last_mu = mu
        self.last_logvar = logvar
        std = np.exp(0.5 * logvar)
        epsilon = np.random.normal(size=mu.shape)
        z = mu + std * epsilon

        self.last_z = z
        self.last_epsilon = epsilon
        return z
