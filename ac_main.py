from fonts.fonts import font_1, font_2, font_3
from fonts.utils import get_all_font_vectors, plot_font_pair, pixel_array_to_char
import json
from autoencoder.neural_network import NeuralNetwork
from utils.activation_functions import relu, logistic, prime_logistic, relu_derivative, prime_tanh, tanh
from utils.optimizers import rosenblatt_optimizer, gradient_descent_optimizer_with_delta, momentum_gradient_descent_optimizer_with_delta, adam_optimizer_with_delta
from utils.error_functions import mean_error, squared_error, mean_squared_error
from plots.latent_space import plot_latent_space
from utils.noise_functions import gaussian_noise, salt_and_pepper_noise
from plots.plots import plot_epoch_network_error
import os

class Parameters():
    def __init__(self, architecture, X_range, hidden_layers_activation_functions, output_layer_activation_function, optimizer, error_functions, epochs, learning_rates):
        self.architecture = architecture
        self.X_range = X_range
        self.hidden_layers_activation_functions = hidden_layers_activation_functions
        self.output_layer_activation_function = output_layer_activation_function
        self.optimizer = optimizer
        self.error_functions = error_functions
        self.epochs = epochs
        self.learning_rates = learning_rates
        

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

    with open("autoencoder_config.json") as f:
        config = json.load(f)

    autoencoder_config = config['autoencoder_configurations']
    problem_config = config['problem_configuration']

    configs = []

    for autoencoder_config in autoencoder_config:
        architecture = autoencoder_config['architecture']
        X_range = autoencoder_config['X_range']
        hidden_layers_activation_functions = autoencoder_config['hidden_layers_activation_functions']
        output_layer_activation_function = autoencoder_config['output_layer_activation_function']
        optimizer = autoencoder_config['optimizer']
        error_functions = autoencoder_config['error_functions']
        epochs = autoencoder_config['epochs']
        learning_rates = autoencoder_config['learning_rates']

        config = Parameters(
            architecture=architecture,
            X_range=X_range,
            hidden_layers_activation_functions=activation_functions_map[hidden_layers_activation_functions],
            output_layer_activation_function=activation_functions_map[output_layer_activation_function],
            optimizer=optimizers_map[optimizer],
            error_functions=error_functions_map[error_functions],
            epochs=epochs,
            learning_rates=learning_rates
        )
        configs.append(config)

    problem_type = problem_config['name']
    font_data = fonts_map[problem_config['font_data']]
    font_init_char = fonts_init_char_map[problem_config['font_data']]

    for config in configs:
        print("##############################################")
        print(f"Running with configuration: {config.__dict__}")
        print(f"Run number: {configs.index(config) + 1} of {len(configs)}")

        network_configuration = config.architecture
        activation_function = config.hidden_layers_activation_functions
        output_layer_activation_function = config.output_layer_activation_function
        optimizer = config.optimizer
        error_function = config.error_functions
        learning_rate = config.learning_rates
        total_epochs = config.epochs
        X_range = config.X_range

        target_values = get_all_font_vectors(font_data)
        target_values = target_values[X_range[0]:X_range[1]]

        if problem_type == "normal":
            X_values = target_values
        elif problem_type == "denoising":
            noise_level = problem_config['denoising_options']['noise_level']
            noise_function = noise_functions_map[problem_config['denoising_options']['noise_function']]
            X_values = [noise_function(vector, noise_level) for vector in target_values]
        elif problem_type == "variational":
            X_values = None

        # Asociación de caracteres a los vectores
        characters = [pixel_array_to_char(array, font_data, font_init_char) for array in target_values]
        print(f"Caracteres asociados: {characters}")

        if (optimizer == gradient_descent_optimizer_with_delta):
            maX_values_error = 0.01            
            neural_network = NeuralNetwork(X_values, network_configuration, activation_function[0], activation_function[1], output_layer_activation_function[0], output_layer_activation_function[1], seed)
            breaking_epoch, training_error = neural_network.backpropagate(X_values, target_values, learning_rate, total_epochs, optimizer, error_function, maX_values_error, is_adam_optimizer= False, activation_function= activation_function[0].__name__, activation_beta=1.0)
            X_values_prime = neural_network.reconstruct_all(X_values)
            plot_latent_space(neural_network, X_values, characters)
            for i in range(len(X_values)):
                plot_font_pair(X_values[i], X_values_prime[i], characters[i])
            for( X_values, X_values_prime) in zip(X_values, X_values_prime):
                print(f"Input: {X_values}")
                print(f"Reconstructed: {X_values_prime}")
            if not os.path.exists("/stats/plots/"):
                os.makedirs("stats/plots/", exist_ok=True)
            
            with open(neural_network.stats.filepath, 'r') as f:
                plot_epoch_network_error(neural_network.stats, neural_network.stats.filepath.replace(".csv", ".png").replace("data/", "plots/"))

        if (optimizer == momentum_gradient_descent_optimizer_with_delta):    
            maX_values_error = 0.1
            alpha = 0.9

            neural_network = NeuralNetwork(X_values, network_configuration, activation_function[0], activation_function[1], output_layer_activation_function[0], output_layer_activation_function[1], seed)
            breaking_epoch, training_error = neural_network.backpropagate(X_values, target_values, learning_rate, total_epochs, optimizer, error_function, maX_values_error, is_adam_optimizer= False, activation_function= activation_function[0].__name__, activation_beta= 1.0, alpha= alpha)
            X_values_prime = neural_network.reconstruct_all(X_values)

            for i in range(len(X_values)):
                plot_font_pair(X_values[i], X_values_prime[i], characters[i])

            for( X_values, X_values_prime) in zip(X_values, X_values_prime):
                print(f"Input: {X_values}")
                print(f"Reconstructed: {X_values_prime}")

            if not os.path.exists("/stats/plots/"):
                os.makedirs("stats/plots/", exist_ok=True)

            with open(neural_network.stats.filepath, 'r') as f:
                plot_epoch_network_error(neural_network.stats, neural_network.stats.filepath.replace(".csv", ".png").replace("data/", "plots/"))
                    
        if (optimizer == adam_optimizer_with_delta):
            maX_values_error = 0.01
            alpha = 0.001
             
            neural_network = NeuralNetwork(X_values, network_configuration, activation_function[0], activation_function[1], output_layer_activation_function[0], output_layer_activation_function[1], seed)
            breaking_epoch, training_error = neural_network.backpropagate(X_values, target_values, learning_rate, total_epochs, optimizer, error_function, maX_values_error, is_adam_optimizer= False, activation_function= activation_function[0].__name__, activation_beta= 0.9, alpha= alpha)
            X_values_prime = neural_network.reconstruct_all(X_values)

            for i in range(len(X_values)):
                plot_font_pair(X_values[i], X_values_prime[i], characters[i])

            for( X_values, X_values_prime) in zip(X_values, X_values_prime):
                print(f"Input: {X_values}")
                print(f"Reconstructed: {X_values_prime}")

            if not os.path.exists("/stats/plots/"):
                os.makedirs("stats/plots/", exist_ok=True)

            with open(neural_network.stats.filepath, 'r') as f:
                plot_epoch_network_error(neural_network.stats, neural_network.stats.filepath.replace(".csv", ".png").replace("data/", "plots/"))
                    