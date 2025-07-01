import numpy as np

class VariationalAutoencoder:
    def __init__(self, encoder_layers_neurons, decoder_layers_neurons, activation_function, activation_derivative, learning_rate=0.001):
        self.encoder_layers_neurons = encoder_layers_neurons
        self.decoder_layers_neurons = decoder_layers_neurons
        self.activation_function = activation_function
        self.prime_activation_function = activation_derivative
        # move to train function
        self.learning_rate = learning_rate

        # we use bias as a row
        self.encoder_weights = []
        for input_to_layer, output_from_layer in zip(encoder_layers_neurons[:-1], encoder_layers_neurons[1:]):
            self.encoder_weights.append(np.random.randn(input_to_layer+ 1, output_from_layer) * 0.1)

        self.decoder_weights = []
        for input_to_layer, output_from_layer in zip(decoder_layers_neurons[:-1], decoder_layers_neurons[1:]):
            self.decoder_weights.append(np.random.randn(input_to_layer+ 1, output_from_layer) * 0.1)

        self.latent_dim = decoder_layers_neurons[0]

    def add_bias_row(self, x):
        return np.hstack([x, np.ones((x.shape[0], 1))])

    def layer_forward(self, x, weights_matrixes):
        a_j_array = [x]
        h_j_array = []
        for w_matrix in weights_matrixes:
            x_with_bias = self.add_bias_row(a_j_array[-1])
            h_j = x_with_bias @ w_matrix

            h_j_array.append(h_j)

            a_j = self.activation_function(h_j)
            a_j_array.append(a_j)
        return a_j_array, h_j_array

    # calculating deltas
    def layer_backward(self, weights, activations, pre_activations, first_delta):
        deltas = [None] * len(weights)

        # receives the first delta considering the network output
        delta = first_delta

        for i in reversed(range(len(weights))):
            act_deriv = self.prime_activation_function(pre_activations[i])
            delta = delta * act_deriv  
            x_with_bias = self.add_bias_row(activations[i])
            deltas[i] = x_with_bias.T @ delta
            if i > 0:
                W_no_bias = weights[i][:-1, :]
                delta = delta @ W_no_bias.T
        return deltas

    def encode(self, x):
        a_j_array, h_j_array = self.layer_forward(x, self.encoder_weights)
        encoder_output = a_j_array[-1]
        # separating into two arrays of latent space dim
        mu = encoder_output[:, :self.latent_dim]
        logvar = encoder_output[:, self.latent_dim:]
        return mu, logvar, a_j_array, h_j_array

    def decode(self, z):
        a_j_array, h_j_array = self.layer_forward(z, self.decoder_weights)
        return a_j_array[-1], a_j_array, h_j_array

    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        epsilon = np.random.randn(*mu.shape)
        return mu + epsilon * std, std, epsilon

    def kl_divergence(self, mu, logvar):
        return -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))

    def feed_forward(self, x):
        mu, logvar, enc_aj_array, enc_hj_array = self.encode(x)
        z, std, epsilon = self.reparameterize(mu, logvar)
        x_hat, dec_aj_array, dec_hj_array = self.decode(z)
        return x_hat, mu, logvar, enc_aj_array, enc_hj_array, dec_aj_array, dec_hj_array, z, std, epsilon

    def backpropagate(self, x, x_hat, mu, logvar, enc_aj_array, enc_hj_array, dec_aj_array, dec_hj_array, z, std, epsilon):
        d_x_hat = 2 * (x_hat - x) / x.shape[0]

        decoder_grads = self.layer_backward(self.decoder_weights, dec_aj_array, dec_hj_array, d_x_hat)

        delta = d_x_hat
        dz = 0
        for j in reversed(range(len(self.decoder_weights))):
            derivative = self.prime_activation_function(dec_hj_array[j])
            delta = delta * derivative
            delta = delta @ self.decoder_weights[j][:-1, :].T

        dz_mu_first_term = delta

        dz_logvar_first_term = dz * (epsilon * std * 0.5)

        dkl_mu = mu
        dkl_logvar = 0.5 * (np.exp(logvar) - 1)

        dj_mu = dz_mu_first_term + dkl_mu
        dj_logvar = dz_logvar_first_term + dkl_logvar

        latent_grad = np.hstack([dj_mu, dj_logvar])

        encoder_grads = self.layer_backward(self.encoder_weights, enc_aj_array, enc_hj_array, latent_grad)

        for i in range(len(self.encoder_weights)):
            self.encoder_weights[i] -= self.learning_rate * encoder_grads[i]
        for i in range(len(self.decoder_weights)):
            self.decoder_weights[i] -= self.learning_rate * decoder_grads[i]

    def train(self, dataset, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            total_kl_loss = 0
            total_reconstruction_loss = 0
            for x in dataset:
                x = x[np.newaxis, :]
                x_hat, mu, logvar, enc_aj_array, enc_hj_array, dec_aj_array, dec_hj_array, z, std, epsilon = self.feed_forward(x)
                kl = self.kl_divergence(mu, logvar)
                #mse
                reconstruction_loss = np.mean((x - x_hat) ** 2) 
                param_lambda = 1
                loss = reconstruction_loss + param_lambda * kl
                total_loss += loss
                total_kl_loss += kl
                total_reconstruction_loss +=reconstruction_loss
                self.backpropagate(x, x_hat, mu, logvar, enc_aj_array, enc_hj_array, dec_aj_array, dec_hj_array, z, std, epsilon)
            print(f"Epoch {epoch + 1} of {epochs}, total_loss: {total_loss:.4f}, reconstruction-loss: {total_reconstruction_loss:.4f}, kl-loss: {total_kl_loss:.4f}")

    def generate_from_specific_samples(self, samples):
        #making sure the dimension is correct based on the latent dimension defined
        for sample in samples:
            if (len(sample) != self.latent_dim):
                raise ValueError(f"{sample} should match the latent dimension: {self.latent_dim}")
        generated_outputs, _, _ = self.decode(samples)
        return generated_outputs

    def generate_from_random_samples(self, samples_num=1):
        z = np.random.randn(samples_num, self.latent_dim)
        generated_outputs, _, _ = self.decode(z)
        return generated_outputs

    def predict(self, x):
        x_hat, _, _, _, _, _, _, _, _, _= self.feed_forward(x)
        return x_hat
