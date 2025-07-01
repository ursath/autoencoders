import matplotlib.pyplot as plt
import numpy as np
import os
from autoencoder.neural_network import NeuralNetwork
from vae.vae import VariationalAutoencoder
from fonts.emoji_utils import plot_font_single_emoji

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
            for i, label in enumerate(labels):
                plt.text(mus[i, 0], mus[i, 1], str(label), fontsize=9, ha='center', va='center')

        plt.title("Espacio latente (2D) de las entradas")
        plt.xlabel("Latente 1")
        plt.ylabel("Latente 2")
        plt.grid(True)
        plt.savefig(f"espacio_latente_para_{text}.png")

def sample_latent_space_grid(neural_network, grid_range=(-3, 3), n_points=10, save_individual=False, save_grid=True, prefix="latent_sample"):
    """
    Samplea el espacio latente en una grilla regular y genera emojis para cada punto.
    
    Args:
        neural_network: El VAE entrenado
        grid_range: Tupla (min, max) del rango del espacio latente
        n_points: Número de puntos por dimensión
        save_individual: Si guardar cada emoji individual
        save_grid: Si guardar la grilla completa
        prefix: Prefijo para los nombres de archivo
    
    Returns:
        latent_points: Array de puntos del espacio latente
        generated_emojis: Array de emojis generados
    """
    # Crear grilla de puntos en el espacio latente
    grid_x = np.linspace(grid_range[0], grid_range[1], n_points)
    grid_y = np.linspace(grid_range[0], grid_range[1], n_points)
    
    # Crear meshgrid y aplanar
    X, Y = np.meshgrid(grid_x, grid_y)
    latent_points = np.column_stack([X.ravel(), Y.ravel()]) 
    
    # Generar emojis para cada punto del espacio latente
    generated_emojis = neural_network.generate_from_specific_samples(latent_points)
    
    # Guardar emojis individuales si se solicita
    if save_individual:
        os.makedirs(f"results/latent_samples/{prefix}", exist_ok=True)
        for idx, (point, emoji) in enumerate(zip(latent_points, generated_emojis)):
            i = idx // n_points
            j = idx % n_points
            filename = f"{prefix}_x{point[0]:.2f}_y{point[1]:.2f}_grid_{i}_{j}"
            plot_font_single_emoji(emoji, f"latent_samples/{prefix}/{filename}")
    
    # Crear y guardar grilla completa
    if save_grid:
        create_latent_space_grid_image(generated_emojis, n_points, prefix, grid_range)
    
    return latent_points, generated_emojis

def create_latent_space_grid_image(generated_emojis, n_points, prefix, grid_range):
    """
    Crea una imagen de grilla mostrando todos los emojis generados del espacio latente.
    """
    fig, axes = plt.subplots(n_points, n_points, figsize=(n_points * 2, n_points * 2))
    
    if n_points == 1:
        axes = np.array([[axes]])
    elif n_points > 1 and axes.ndim == 1:
        axes = axes.reshape(1, -1)
    
    for idx, emoji in enumerate(generated_emojis):
        i = idx // n_points
        j = idx % n_points
        
        # Redimensionar emoji si está aplanado
        if emoji.ndim == 1:
            emoji_img = emoji.reshape(16, 16, 4)
        else:
            emoji_img = emoji
        
        # Asegurar valores en [0, 1]
        emoji_img = np.clip(emoji_img, 0, 1)
        
        # Mostrar emoji
        axes[i, j].imshow(emoji_img)
        axes[i, j].axis('off')
        
        # Agregar coordenadas como título
        x_coord = grid_range[0] + (grid_range[1] - grid_range[0]) * j / (n_points - 1)
        y_coord = grid_range[1] - (grid_range[1] - grid_range[0]) * i / (n_points - 1)  # Invertido para que y crezca hacia arriba
        axes[i, j].set_title(f"({x_coord:.1f}, {y_coord:.1f})", fontsize=8)
    
    plt.tight_layout()
    
    # Guardar imagen
    os.makedirs("results/latent_grids", exist_ok=True)
    output_path = f"results/latent_grids/{prefix}_grid_{n_points}x{n_points}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Grilla del espacio latente guardada en: {output_path}")

def sample_latent_space_random(neural_network, n_samples=50, grid_range=(-3, 3), prefix="random_sample"):
    """
    Samplea puntos aleatorios del espacio latente.
    
    Args:
        neural_network: El VAE entrenado
        n_samples: Número de muestras aleatorias
        grid_range: Rango para el muestreo uniforme
        prefix: Prefijo para los nombres de archivo
    
    Returns:
        latent_points: Array de puntos aleatorios del espacio latente
        generated_emojis: Array de emojis generados
    """
    # Generar puntos aleatorios en el espacio latente
    latent_points = np.random.uniform(grid_range[0], grid_range[1], (n_samples, 2))
    
    # Generar emojis
    generated_emojis = neural_network.generate_from_specific_samples(latent_points)
    
    # Guardar algunos ejemplos
    os.makedirs(f"results/random_samples/{prefix}", exist_ok=True)
    for idx, (point, emoji) in enumerate(zip(latent_points[:10], generated_emojis[:10])):  # Solo primeros 10
        filename = f"{prefix}_random_{idx}_x{point[0]:.2f}_y{point[1]:.2f}"
        plot_font_single_emoji(emoji, f"random_samples/{prefix}/{filename}")
    
    return latent_points, generated_emojis