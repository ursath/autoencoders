import matplotlib.pyplot as plt
import numpy as np
import os
from autoencoder.neural_network import NeuralNetwork

def plot_latent_space(neural_network: NeuralNetwork, X, labels):
    latent_vectors = []
    for input_vector in X:
        z = neural_network.encode_to_latent_space(input_vector)
        latent_vectors.append(z)

    latent_vectors = np.array(latent_vectors)
    fig = plt.figure(figsize=(8,6))
    for i, point in enumerate(latent_vectors):
        plt.scatter(point[0], point[1], marker='o')
        plt.text(point[0] + 0.01, point[1] + 0.01, labels[i], fontsize=9)
    plt.title("Espacio latente (2D) de las letras")
    plt.xlabel("Latente 1")
    plt.ylabel("Latente 2")
    plt.grid(True)

    output_path = os.path.join("results", f"latent_space_2D.png")
    plt.savefig(output_path)
    plt.close(fig) 
    