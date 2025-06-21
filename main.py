from fonts.font import font_3
from fonts.utils import get_all_font_vectors, plot_font_pair
import numpy as np
import json
from typing import List
from autoencoder.neural_network import NeuralNetwork
from utils.activation_functions import relu, logistic, prime_logistic, relu_derivative, prime_tanh, tanh
from utils.optimizers import rosenblatt_optimizer, gradient_descent_optimizer_with_delta, momentum_gradient_descent_optimizer_with_delta, adam_optimizer_with_delta
from utils.error_functions import mean_error, squared_error

if __name__ == "__main__":
    
    seed:int = 43

    activation_functions_map = {
        "relu": (relu, relu_derivative),
        "logistic": (logistic, prime_logistic),
        "tanh": (tanh, prime_tanh)
    }

    error_functions_map = {
        "squared_error": squared_error,
        "mean_error": mean_error
    }

    with open("config.json") as f:
        config = json.load(f)

    autoencoder_config = config['autoencoder']

    network_configurations = autoencoder_config['architecture']
    activation_functions = [activation_functions_map[name] for name in autoencoder_config['hidden_layers_activation_functions']]
    output_layer_activation_function = autoencoder_config['output_layer_activation_function']
    error_functions = [error_functions_map[name] for name in autoencoder_config['error_functions']]
    epochs = autoencoder_config['epochs']
    learning_rates = autoencoder_config['learning_rates']
    X_range = autoencoder_config['X_range']
    
    def train_and_evaluate(optimizer):
        X = get_all_font_vectors(font_3)
        X = X[X_range[0]:X_range[1]]

        if(optimizer == gradient_descent_optimizer_with_delta):
            max_error = 0.01
            for network_configuration in network_configurations:
                for activation_function in activation_functions:
                        for error_function in error_functions:
                            for learning_rate in learning_rates:
                                for total_epochs in epochs:

                                    neural_network = NeuralNetwork(X, network_configuration, activation_function[0], activation_function[1], seed)
                                    breaking_epoch, training_error = neural_network.backpropagate(X, learning_rate, total_epochs, optimizer, error_function, max_error, is_adam_optimizer= False, activation_function= activation_function[0].__name__, activation_beta=1.0)
                                    X_prime = neural_network.reconstruct_all(X)
                                    
                                    for i in range(len(X)):
                                        plot_font_pair(X[i], X_prime[i])

                                    for( X, X_prime) in zip(X, X_prime):
                                        print(f"Input: {X}")
                                        print(f"Reconstructed: {X_prime}")

        if(optimizer == adam_optimizer_with_delta):
            max_error = 1.0
            alpha = 0.9
            for network_configuration in network_configurations:
                for activation_function in activation_functions:
                        for error_function in error_functions:
                            for learning_rate in learning_rates:
                                for total_epochs in epochs:
                                    neural_network = NeuralNetwork(X, network_configuration, activation_function[0], activation_function[1], seed)
                                    breaking_epoch, training_error = neural_network.backpropagate(X, learning_rate, total_epochs, optimizer, error_function, max_error, is_adam_optimizer= False, activation_function= activation_function[0].__name__, activation_beta= 1.0, alpha= alpha)
                                    X_prime = neural_network.reconstruct_all(X)

                                    for i in range(len(X)):
                                        plot_font_pair(X[i], X_prime[i])

                                    for( X, X_prime) in zip(X, X_prime):
                                        print(f"Input: {X}")
                                        print(f"Reconstructed: {X_prime}")


    ###################################################### RUN ##################################################################

    train_and_evaluate(gradient_descent_optimizer_with_delta)
    #train_and_evaluate(momentum_gradient_descent_optimizer_with_delta)
    #train_and_evaluate(adam_optimizer_with_delta)

