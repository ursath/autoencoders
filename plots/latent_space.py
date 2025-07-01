import matplotlib.pyplot as plt
import numpy as np
import os
from autoencoder.neural_network import NeuralNetwork
from vae.vae import VariationalAutoencoder

folder_path = "results"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def plot_latent_space(neural_network: NeuralNetwork, X, labels):
    latent_vectors = []
    for input_vector in X:
        z = neural_network.encode_to_latent_space(input_vector)
        latent_vectors.append(z)

    latent_vectors = np.array(latent_vectors)
    fig = plt.figure(figsize=(8,6))
    for i, point in enumerate(latent_vectors):
        plt.scatter(point[0], point[1], marker='o')
        plt.text(point[0], point[1] + 0.005, labels[i], fontsize=9, ha='center', va='bottom')
    plt.title("Espacio latente (2D) de las letras")
    plt.xlabel("Latente 1")
    plt.ylabel("Latente 2")
    plt.grid(True)

    output_path = os.path.join("results", f"latent_space_2D.png")
    plt.savefig(output_path)
    plt.close(fig) 
    
def plot_latent_space_2d_scatter(neural_network, data, labels=None, text=""):
        mus = []
        for x in data:
            x = x[np.newaxis, :]
            mu, _, _, _ = neural_network.encode(x)
            mus.append(mu[0])

        mus = np.array(mus)

        plt.figure(figsize=(8, 6))
        plt.scatter(mus[:, 0], mus[:, 1], alpha=0.6, c='cornflowerblue')

        if labels is not None:
            n_points = min(len(mus), len(labels))
            for i in range(n_points):
                plt.text(mus[i, 0], mus[i, 1], str(labels[i]), fontsize=9, ha='center', va='center')

        plt.title("Espacio latente (2D) de las entradas")
        plt.xlabel("Latente 1")
        plt.ylabel("Latente 2")
        plt.grid(True)
        plt.savefig(f"espacio_latente_para_{text}.png")