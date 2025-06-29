from fonts.fonts import font_1, font_2, font_3, emojis
from fonts.utils import get_all_font_vectors, pixel_array_to_char, plot_font_grid, plot_font_single
import json
from autoencoder.neural_network import NeuralNetwork
from vae.variational_neural_network import VariationalNeuralNetwork
from utils.activation_functions import relu, logistic, prime_logistic, relu_derivative, prime_tanh, tanh
from utils.optimizers import rosenblatt_optimizer, gradient_descent_optimizer_with_delta, momentum_gradient_descent_optimizer_with_delta, adam_optimizer_with_delta
from utils.error_functions import mean_error, squared_error, mean_squared_error
from plots.latent_space import plot_latent_space
from utils.noise_functions import gaussian_noise, salt_and_pepper_noise
from plots.plots import plot_epoch_network_error
from utils.generate_character import generate_new_character_and_plot
from fonts.emoji_utils import get__all_font_vectors_emoji, plot_font_grid_emoji
import numpy as np
import os

if __name__ == "__main__":
    
    seed:int = 43

    # Definición de mapeos

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

    noise_functions_map = {
        "gaussian": gaussian_noise,
        "salt_and_pepper": salt_and_pepper_noise
    }

    fonts_map = {
        "font_1": font_1,
        "font_2": font_2,
        "font_3": font_3
    }

    fonts_init_char_map = {
        "font_1": 0x20,
        "font_2": 0x40,
        "font_3": 0x60,
    }


    # Carga de la configuración desde el archivo JSON

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

    problem_config = config['problem']
    problem_type = problem_config['name']
    font_data = fonts_map[problem_config['font_data']]
    font_init_char = fonts_init_char_map[problem_config['font_data']]


    # Definición de X_values y target_values según el tipo de problema

    target_values = get_all_font_vectors(font_data)
    target_values = target_values[X_range[0]:X_range[1]]
    generate_new_character = False

    if problem_type == "normal":
        X_values = target_values
    elif problem_type == "generate":
        X_values = target_values
        generate_new_character = True
    elif problem_type == "denoising":
        noise_level = problem_config['denoising_options']['noise_level']
        noise_function = noise_functions_map[problem_config['denoising_options']['noise_function']]
        X_values = [noise_function(vector, noise_level) for vector in target_values]
    elif problem_type == "variational":
        X_values = target_values


    # Asociación de caracteres a los vectores

    characters = [pixel_array_to_char(array, font_data, font_init_char) for array in target_values]
    print(f"Caracteres asociados: {characters}")

    # Inicialización de la red neuronal y entrenamiento
    if problem_type == "normal" or problem_type == "denoising":
        if(optimizer == gradient_descent_optimizer_with_delta):
            maX_values_error = 0.01
            for network_configuration in network_configurations:
                for activation_function in activation_functions:
                        for error_function in error_functions:
                            for learning_rate in learning_rates:
                                for total_epochs in epochs:
                                   
                                    neural_network = NeuralNetwork(X_values, network_configuration, activation_function[0], activation_function[1], output_layer_activation_function[0][0], output_layer_activation_function[0][1], seed)
                                    breaking_epoch, training_error = neural_network.backpropagate(X_values, target_values, learning_rate, total_epochs, optimizer, error_function, maX_values_error, is_adam_optimizer= False, activation_function= activation_function[0].__name__, activation_beta=1.0)
                                    X_values_prime = neural_network.reconstruct_all(X_values)

                                    plot_latent_space(neural_network, X_values, characters)
                                    
                                    if(generate_new_character):
                                        generate_new_character_and_plot(neural_network)

                                    plot_font_grid(X_values, X_values_prime)

                                    if not os.path.exists("/stats/plots/"):
                                        os.makedirs("stats/plots/", exist_ok=True)

        if(optimizer == momentum_gradient_descent_optimizer_with_delta):
            maX_values_error = 0.01
            alpha = 0.9
            for network_configuration in network_configurations:
                for activation_function in activation_functions:
                        for error_function in error_functions:
                            for learning_rate in learning_rates:
                                for total_epochs in epochs:
                                    neural_network = NeuralNetwork(X_values, network_configuration, activation_function[0], activation_function[1], output_layer_activation_function[0][0], output_layer_activation_function[0][1], seed)
                                    breaking_epoch, training_error = neural_network.backpropagate(X_values, target_values, learning_rate, total_epochs, optimizer, error_function, maX_values_error, is_adam_optimizer= False, activation_function= activation_function[0].__name__, activation_beta= 1.0, alpha= alpha)
                                    X_values_prime = neural_network.reconstruct_all(X_values)

                                    plot_latent_space(neural_network, X_values, characters)

                                    if(generate_new_character):
                                        generate_new_character_and_plot(neural_network)
                                    
                                    plot_font_grid(X_values, X_values_prime)

                                    
        if(optimizer == adam_optimizer_with_delta):
            maX_values_error = 0.01
            alpha = 0.0001
            for network_configuration in network_configurations:
                for activation_function in activation_functions:
                        for error_function in error_functions:
                            for learning_rate in learning_rates:
                                for total_epochs in epochs:
                                    neural_network = NeuralNetwork(X_values, network_configuration, activation_function[0], activation_function[1], output_layer_activation_function[0][0], output_layer_activation_function[0][1],optimizer, seed)
                                    breaking_epoch, training_error = neural_network.backpropagate(X_values, target_values, learning_rate, total_epochs, optimizer, error_function, maX_values_error, is_adam_optimizer= False, activation_function= activation_function[0].__name__, activation_beta= 0.9, alpha= alpha)
                                    X_values_prime = neural_network.reconstruct_all(X_values)

                                    plot_latent_space(neural_network, X_values, characters)

                                    if(generate_new_character):
                                        generate_new_character_and_plot(neural_network)
                                    
                                    plot_font_grid(X_values, X_values_prime)

    emoji_values = get__all_font_vectors_emoji(emojis)                              
    if problem_type == "variational":    
        maX_values_error = 0.01
        alpha = 0.0001
        for network_configuration in network_configurations:
            for activation_function in activation_functions:
                for error_function in error_functions:
                    for learning_rate in learning_rates:
                        for total_epochs in epochs:
                            neural_network = VariationalNeuralNetwork(X_values, network_configuration, activation_function[0], activation_function[1], output_layer_activation_function[0][0], output_layer_activation_function[0][1], "adam_optimizer_with_delta", seed)
                            breaking_epoch, training_error = neural_network.backpropagate(X_values, target_values, alpha, total_epochs, optimizer, error_function, maX_values_error, is_adam_optimizer= False, activation_function= activation_function[0].__name__)
                            emoji_values_prime = neural_network.reconstruct_all(emoji_values)

                            plot_font_grid_emoji(emoji_values, emoji_values_prime)

                            for( X_values, X_values_prime) in zip(X_values, X_values_prime):
                                print(f"Input: {X_values}")
                                print(f"Reconstructed: {X_values_prime}")

