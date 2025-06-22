import numpy as np
from typing import List
from .layer import Layer
from utils.activation_functions import ActivationFunctionType
from utils.error_functions import ErrorFunctionType
from utils.optimizers import OptimizerFunctionType
from utils.stats import Statistics

class NeuralNetwork:
    # Note: hidden_layers_neuron_amounts actually means all layers (not just hidden layers)
    def __init__(self, x_values:List[List[int]], hidden_layers_neuron_amounts:List[int],
                 activation_function:ActivationFunctionType, prime_activation_function:ActivationFunctionType,
                 output_layer_activation_function:ActivationFunctionType, output_layer_prime_activation_function:ActivationFunctionType, seed:int):

        self.seed = seed
        self.layers = []
        self.x_values = x_values

        filename = (
            f"layers_{'-'.join(map(str, hidden_layers_neuron_amounts))}_"
            f"act_{activation_function.__name__}_"
            f"outact_{output_layer_activation_function.__name__}_"
            f"seed_{seed}"
        )

        self.stats = Statistics(filename)

        input_size = len(x_values[0])
        previous_layer_neurons = input_size

        # Hidden layers
        for current_layer_neuron_amount in hidden_layers_neuron_amounts[:-1]: 
            layer = Layer(
                num_inputs=previous_layer_neurons,                
                num_previous_layer_neurons=previous_layer_neurons,
                num_neurons=current_layer_neuron_amount,
                activation_function=activation_function,
                prime_activation_function=prime_activation_function,
                seed=seed
            )
            self.layers.append(layer)
            previous_layer_neurons = current_layer_neuron_amount 

        # Output layer
        output_layer = Layer(
            num_inputs=previous_layer_neurons,
            num_previous_layer_neurons=previous_layer_neurons,
            num_neurons=len(x_values[0]),  
            activation_function=output_layer_activation_function,
            prime_activation_function=output_layer_prime_activation_function,
            seed=seed
        )

        self.layers.append(output_layer)

        self.weight_matrixes = [layer.weights_matrix for layer in self.layers]


    def predict(self, input_values:List[int], beta:float=1.0):
        a_j_vector = input_values
        for layer in self.layers:
            a_j_vector = layer.forward(a_j_vector, beta)
        return a_j_vector
    
    def reconstruct_all(self, input_values:List[int], beta=1.0):
        return np.array([self.predict(x, beta) for x in input_values])
    
    def encode_to_latent_space(self, input_values: List[int], beta: float = 1.0) -> np.ndarray:
        a_j_vector = input_values
        for layer in self.layers:
            a_j_vector = layer.forward(a_j_vector, beta)
            if len(layer.a_j_values) == 2:  # llego a la capa latente
                return a_j_vector
    
    def backpropagate(self, input_values:List[List[int]], target_values:List[List[int]], learning_rate:float, epochs:int, optimizer:OptimizerFunctionType, error_function:ErrorFunctionType, max_acceptable_error:float, is_adam_optimizer=False, activation_function= "", activation_beta:float= 1.0, alpha:float= 0.0):
        m_k_matrixes = []
        v_k_matrixes = []
        prev_delta_w_matrixes = []
        for epoch in range(epochs):
            for input_vector, target_vector in zip(input_values, target_values):
                prediction = self.predict(input_vector)

                basic_error = target_vector - prediction
                
                reverse_layers = self.layers[::-1]

                output_layer = reverse_layers[0]
                output_delta = basic_error * output_layer.prime_activation_function(output_layer.h_j_values, activation_beta)
                layer_deltas = [output_delta]

                # we calculate all deltas before updating weights
                for layer_index in range(1, len(reverse_layers)):
                    current_layer = reverse_layers[layer_index]
                    next_layer = reverse_layers[layer_index-1]

                    # first we try doing it without bias
                    next_weights = next_layer.weights_matrix[:, 1:]  

                    propagated_error = np.dot(layer_deltas[0], next_weights)  
                    current_delta = current_layer.prime_activation_function(current_layer.h_j_values, activation_beta) * propagated_error

                    layer_deltas.insert(0, current_delta)

            #batch
                for layer_index, layer in enumerate(self.layers):
                    delta = layer_deltas[layer_index]
                    if (layer_index == 0):
                        input_to_layer = input_vector
                    else:
                        input_to_layer = self.layers[layer_index-1].a_j_values

                    if optimizer.__name__ == 'adam_optimizer_with_delta':
                        if (epoch == 0):
                            m_k_matrix = np.zeros((len(delta), len(input_to_layer)))
                            m_k_matrixes.append(m_k_matrix)
                            v_k_matrix = np.zeros((len(delta), len(input_to_layer)))
                            v_k_matrixes.append(v_k_matrix)

                        for j in range(len(delta)):
                            for i in range(len(input_to_layer)):
                                delta_w, m_k, v_k = optimizer(delta[j], input_to_layer[i], alpha, 0.9, 0.999, 1e-6, m_k_matrixes[layer_index][j][i], v_k_matrixes[layer_index][j][i],epoch)
                                layer.weights_matrix[j][i+1] += delta_w 
                                m_k_matrixes[layer_index][j][i] = m_k
                                v_k_matrixes[layer_index][j][i] = v_k
                    else:
                        if optimizer.__name__ == 'momentum_gradient_descent_optimizer_with_delta':
                            if (epoch == 0):
                                prev_delta_w_matrix = np.zeros((len(delta), len(input_to_layer)))
                                prev_delta_w_matrixes.append(prev_delta_w_matrix)

                            for j in range(len(delta)):
                                for i in range(len(input_to_layer)):
                                    delta_w = optimizer(learning_rate, delta[j], input_to_layer[i], prev_delta_w_matrixes[layer_index][j][i], alpha)
                                    layer.weights_matrix[j][i+1] += delta_w 
                                    prev_delta_w_matrixes[layer_index][j][i] = float(delta_w)
                            
                        else:
                            for j in range(len(delta)):
                                for i in range(len(input_to_layer)):
                                    delta_w = optimizer(learning_rate, delta[j], input_to_layer[i], alpha)
                                    layer.weights_matrix[j][i+1] += delta_w 
            
            errors = []
            for input_vector, target_vector in zip(input_values, target_values):
                prediction = self.predict(input_vector)
                basic_error = target_vector - prediction
                errors.append(basic_error)
            
            network_error = error_function(np.array(errors))
            self.stats.write(f"{epoch+1},{network_error}")
            print(f"Epoch {epoch+1}/{epochs}, Network Error: {network_error}")

            if network_error < max_acceptable_error:
                return epoch+1, network_error
            
        return epochs, network_error