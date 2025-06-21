from fonts.font import font_3
from fonts.utils import get_all_font_vectors, plot_font_pair, pixel_array_to_char
import numpy as np
import json
from typing import List
from autoencoder.neural_network import NeuralNetwork
from utils.activation_functions import relu, logistic, prime_logistic, relu_derivative, prime_tanh, tanh
from utils.optimizers import rosenblatt_optimizer, gradient_descent_optimizer_with_delta, momentum_gradient_descent_optimizer_with_delta, adam_optimizer_with_delta
from utils.error_functions import mean_error, squared_error, mean_squared_error
from plots.latent_space import plot_latent_space

if __name__ == "__main__":
    
    seed:int = 43

    activation_functions_map = {
        "relu": (relu, relu_derivative),
        "logistic": (logistic, prime_logistic),
        "tanh": (tanh, prime_tanh)
    }

    error_functions_map = {
        "squared_error": squared_error,
        "mean_squared_error": mean_squared_error,
        "mean_error": mean_error
    }

    optimizers_map = {
        "rosenblatt": rosenblatt_optimizer,
        "gradient_descent": gradient_descent_optimizer_with_delta,
        "momentum": momentum_gradient_descent_optimizer_with_delta,
        "adam": adam_optimizer_with_delta
    }

    with open("config.json") as f:
        config = json.load(f)

    autoencoder_config = config['autoencoder']

    network_configurations = autoencoder_config['architecture']
    activation_functions = [activation_functions_map[name] for name in autoencoder_config['hidden_layers_activation_functions']]
    output_layer_activation_function = [activation_functions_map[name] for name in autoencoder_config['output_layer_activation_function']]
    optimizer = optimizers_map[autoencoder_config['optimizer']]
    error_functions = [error_functions_map[name] for name in autoencoder_config['error_functions']]
    epochs = autoencoder_config['epochs']
    learning_rates = autoencoder_config['learning_rates']
    X_range = autoencoder_config['X_range']

    font_data = font_3
    X_values = get_all_font_vectors(font_data)
    X_values = X_values[X_range[0]:X_range[1]]
    characters = [pixel_array_to_char(array, font_data) for array in X_values]
       

    if(optimizer == gradient_descent_optimizer_with_delta):
        maX_values_error = 0.01
        for network_configuration in network_configurations:
            for activation_function in activation_functions:
                    for error_function in error_functions:
                        for learning_rate in learning_rates:
                            for total_epochs in epochs:

                                neural_network = NeuralNetwork(X_values, network_configuration, activation_function[0], activation_function[1], output_layer_activation_function[0][0], output_layer_activation_function[0][1], seed)
                                breaking_epoch, training_error = neural_network.backpropagate(X_values, learning_rate, total_epochs, optimizer, error_function, maX_values_error, is_adam_optimizer= False, activation_function= activation_function[0].__name__, activation_beta=1.0)
                                X_values_prime = neural_network.reconstruct_all(X_values)
                                
                                plot_latent_space(neural_network, X_values, characters)

                                for i in range(len(X_values)):
                                    plot_font_pair(X_values[i], X_values_prime[i], characters[i])

                                for( X_values, X_values_prime) in zip(X_values, X_values_prime):
                                    print(f"Input: {X_values}")
                                    print(f"Reconstructed: {X_values_prime}")

    if(optimizer == momentum_gradient_descent_optimizer_with_delta):
        maX_values_error = 0.1
        alpha = 0.9
        for network_configuration in network_configurations:
            for activation_function in activation_functions:
                    for error_function in error_functions:
                        for learning_rate in learning_rates:
                            for total_epochs in epochs:
                                neural_network = NeuralNetwork(X_values, network_configuration, activation_function[0], activation_function[1], seed)
                                breaking_epoch, training_error = neural_network.backpropagate(X_values, learning_rate, total_epochs, optimizer, error_function, maX_values_error, is_adam_optimizer= False, activation_function= activation_function[0].__name__, activation_beta= 1.0, alpha= alpha)
                                for i in range(len(X_values)):
                                    plot_font_pair(X_values[i], X_values_prime[i], font_data)

                                for( X_values, X_values_prime) in zip(X_values, X_values_prime):
                                    print(f"Input: {X_values}")
                                    print(f"Reconstructed: {X_values_prime}")
                                
    if(optimizer == adam_optimizer_with_delta):
        maX_values_error = 0.01
        alpha = 0.001
        for network_configuration in network_configurations:
            for activation_function in activation_functions:
                    for error_function in error_functions:
                        for learning_rate in learning_rates:
                            for total_epochs in epochs:
                                neural_network = NeuralNetwork(X_values, network_configuration, activation_function[0], activation_function[1], seed)
                                breaking_epoch, training_error = neural_network.backpropagate(X_values, learning_rate, total_epochs, optimizer, error_function, maX_values_error, is_adam_optimizer= False, activation_function= activation_function[0].__name__, activation_beta= 0.9, alpha= alpha)
                                X_values_prime = neural_network.reconstruct_all(X_values)


                                for i in range(len(X_values)):
                                    plot_font_pair(X_values[i], X_values_prime[i], font_data)

                                for( X_values, X_values_prime) in zip(X_values, X_values_prime):
                                    print(f"Input: {X_values}")
                                    print(f"Reconstructed: {X_values_prime}")

