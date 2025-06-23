import numpy as np
from typing import List
from .variational_layer import Layer,LatentSpaceLayer
from utils.activation_functions import ActivationFunctionType
from utils.activation_functions import identity, prime_identity
from utils.error_functions import ErrorFunctionType
from utils.optimizers import OptimizerFunctionType
from utils.stats import Statistics

class VariationalNeuralNetwork:
    # Note: hidden_layers_neuron_amounts actually means all layers (not just hidden layers)
    def __init__(self, x_values:List[List[int]], hidden_layers_neuron_amounts:List[int],
                 activation_function:ActivationFunctionType, prime_activation_function:ActivationFunctionType,
                 output_layer_activation_function:ActivationFunctionType, output_layer_prime_activation_function:ActivationFunctionType, optimizer:str, seed:int):

        self.seed = seed
        self.encoder_layers = []
        self.decoder_layers = []
        self.x_values = x_values


        latent_layer_index = np.floor(len(hidden_layers_neuron_amounts) / 2).astype(int)
        # Creation of spetial layer for media and variance
        self.mu_layer = Layer(
            num_inputs = hidden_layers_neuron_amounts[latent_layer_index - 1],
            num_previous_layer_neurons = hidden_layers_neuron_amounts[latent_layer_index - 1],
            num_neurons= 2 ,
            activation_function=identity,
            prime_activation_function=prime_identity,
            seed=seed
        )

        self.variance_layer = Layer(
            num_inputs = hidden_layers_neuron_amounts[latent_layer_index - 1],
            num_previous_layer_neurons = hidden_layers_neuron_amounts[latent_layer_index - 1],
            num_neurons= 2 ,
            activation_function=identity,
            prime_activation_function=prime_identity,
            seed=seed
        )

        # Creation of the latent space layer
        self.latent_space_layer = LatentSpaceLayer()

        filename = (
            f"layers_{'-'.join(map(str, hidden_layers_neuron_amounts))}_"
            f"act_{activation_function.__name__}_"
            f"outact_{output_layer_activation_function.__name__}_"
            f"seed_{seed}_"
            f"{optimizer}"
        )

        self.stats = Statistics(filename)

        input_size = len(x_values[0])
        previous_layer_neurons = input_size

        # Encoder layers
        for current_layer_neuron_amount in hidden_layers_neuron_amounts[:latent_layer_index]: 
            
            layer = Layer(
                num_inputs=previous_layer_neurons,                
                num_previous_layer_neurons=previous_layer_neurons,
                num_neurons=current_layer_neuron_amount,
                activation_function=activation_function,
                prime_activation_function=prime_activation_function,
                seed=seed
            )
                
            self.encoder_layers.append(layer)
            previous_layer_neurons = current_layer_neuron_amount 

        latent_dim = hidden_layers_neuron_amounts[latent_layer_index]
        previous_layer_neurons = latent_dim

        # Decoder layers
        for current_layer_neuron_amount in hidden_layers_neuron_amounts[latent_layer_index + 1:-1]:
            layer = Layer(
                num_inputs=previous_layer_neurons,                
                num_previous_layer_neurons=previous_layer_neurons,
                num_neurons=current_layer_neuron_amount,
                activation_function=activation_function,
                prime_activation_function=prime_activation_function,
                seed=seed
            )
                
            self.decoder_layers.append(layer)
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

        self.decoder_layers.append(output_layer)

        self.encoder_weight_matrixes = [layer.weights_matrix for layer in self.encoder_layers]
        self.decoder_weight_matrixes = [layer.weights_matrix for layer in self.decoder_layers]


    def predict(self, input_values:List[int], beta:float=1.0):
        a_j_vector = input_values
        for layer in self.encoder_layers:
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
    
    def backpropagate(self, input_values, target_values, learning_rate, epochs, optimizer, error_function, max_acceptable_error, is_adam_optimizer=False, activation_function="", activation_beta=1.0, alpha=0.0):
        m_k_encoder, v_k_encoder, m_k_decoder, v_k_decoder = [], [], [], []
        prev_dw_encoder, prev_dw_decoder = [], []
        m_k_mu = v_k_mu = m_k_var = v_k_var = None

        for epoch in range(epochs):
            total_recon_loss = 0
            total_kl_loss = 0

            for x, y in zip(input_values, target_values):
                # --- Forward Pass ---
                x_encoded = x
                for layer in self.encoder_layers:
                    x_encoded = layer.forward(x_encoded)

                mu = self.mu_layer.forward(x_encoded)
                logvar = self.variance_layer.forward(x_encoded)

                z = self.latent_space_layer.forward(mu, logvar)
                epsilon = self.latent_space_layer.last_epsilon


                x_hat = z
                for layer in self.decoder_layers:
                    x_hat = layer.forward(x_hat)

                recon_error = y - x_hat
                kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))

                total_recon_loss += np.sum(recon_error ** 2)
                total_kl_loss += kl_loss

                # --- Backward Pass ---
                # Decoder
                delta = recon_error
                for layer in reversed(self.decoder_layers):
                    delta = layer.backward(delta)

                grad_z = delta
                std = np.exp(0.5 * logvar)
                dz_dmu = 1
                dz_dlogvar = 0.5 * std * epsilon

                kl_mu_grad = mu
                kl_logvar_grad = 0.5 * (np.exp(logvar) - 1)

                delta_mu = grad_z * dz_dmu + kl_mu_grad
                delta_logvar = grad_z * dz_dlogvar + kl_logvar_grad

                delta_mu = self.mu_layer.backward(delta_mu)
                delta_logvar = self.variance_layer.backward(delta_logvar)

                delta = delta_mu + delta_logvar
                for layer in reversed(self.encoder_layers):
                    delta = layer.backward(delta)

                # --- Weight Updates ---
                # Encoder
                for i, layer in enumerate(self.encoder_layers):
                    input_to_layer = x if i == 0 else self.encoder_layers[i - 1].a_j_values
                    if optimizer.__name__ == 'adam_optimizer_with_delta':
                        if epoch == 0:
                            m_k_encoder.append(np.zeros((len(layer.last_delta), len(input_to_layer) + 1)))
                            v_k_encoder.append(np.zeros((len(layer.last_delta), len(input_to_layer) + 1)))
                        layer.update_weights(learning_rate, optimizer, input_to_layer, m_k_encoder[i], v_k_encoder[i], epoch, alpha)
                    elif optimizer.__name__ == 'momentum_gradient_descent_optimizer_with_delta':
                        if epoch == 0:
                            prev_dw_encoder.append(np.zeros((len(layer.last_delta), len(input_to_layer) + 1)))
                        layer.update_weights(learning_rate, optimizer, input_to_layer, prev_dw=prev_dw_encoder[i], alpha=alpha)
                    else:
                        layer.update_weights(learning_rate, optimizer, input_to_layer, alpha=alpha)

                # Decoder
                for i, layer in enumerate(self.decoder_layers):
                    input_to_layer = z if i == 0 else self.decoder_layers[i - 1].a_j_values
                    if optimizer.__name__ == 'adam_optimizer_with_delta':
                        if epoch == 0:
                            m_k_decoder.append(np.zeros((len(layer.last_delta), len(input_to_layer) + 1)))
                            v_k_decoder.append(np.zeros((len(layer.last_delta), len(input_to_layer) + 1)))
                        layer.update_weights(learning_rate, optimizer, input_to_layer, m_k_decoder[i], v_k_decoder[i], epoch, alpha)
                    elif optimizer.__name__ == 'momentum_gradient_descent_optimizer_with_delta':
                        if epoch == 0:
                            prev_dw_decoder.append(np.zeros((len(layer.last_delta), len(input_to_layer) + 1)))
                        layer.update_weights(learning_rate, optimizer, input_to_layer, prev_dw=prev_dw_decoder[i], alpha=alpha)
                    else:
                        layer.update_weights(learning_rate, optimizer, input_to_layer, alpha=alpha)

                # Mu y Var
                input_to_mu = self.encoder_layers[-1].a_j_values
                input_to_logvar = input_to_mu

                if optimizer.__name__ == 'adam_optimizer_with_delta':
                    if epoch == 0:
                        m_k_mu = np.zeros((len(self.mu_layer.last_delta), len(input_to_mu) + 1))
                        v_k_mu = np.zeros((len(self.mu_layer.last_delta), len(input_to_mu) + 1))
                        m_k_var = np.zeros((len(self.variance_layer.last_delta), len(input_to_logvar) + 1))
                        v_k_var = np.zeros((len(self.variance_layer.last_delta), len(input_to_logvar) + 1))

                    self.mu_layer.update_weights(learning_rate, optimizer, input_to_mu, m_k=m_k_mu, v_k=v_k_mu, epoch=epoch, alpha=alpha)
                    self.variance_layer.update_weights(learning_rate, optimizer, input_to_logvar, m_k=m_k_var, v_k=v_k_var, epoch=epoch, alpha=alpha)
                else:
                    self.mu_layer.update_weights(learning_rate, optimizer, input_to_mu, alpha=alpha)
                    self.variance_layer.update_weights(learning_rate, optimizer, input_to_logvar, alpha=alpha)

            avg_recon = total_recon_loss / len(input_values)
            avg_kl = total_kl_loss / len(input_values)
            total_loss = avg_recon + avg_kl
            self.stats.write(f"{epoch+1},{avg_recon},{avg_kl},{total_loss}")
            print(f"Epoch {epoch + 1}, Recon Loss: {avg_recon:.6f}, KL Loss: {avg_kl:.6f}, Total Loss: {total_loss:.6f}")

            if total_loss < max_acceptable_error:
                print(f"Early stopping at epoch {epoch + 1} with total loss {total_loss:.6f}")
                return epoch + 1, total_loss

        return epochs, total_loss

