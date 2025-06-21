import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def to_bin_array(encoded_caracter):
    bin_array = np.zeros((7, 5), dtype=int)
    for row in range(0, 7):
        current_row = encoded_caracter[row]
        for col in range(0, 5):
            bin_array[row][4-col] = current_row & 1
            current_row >>= 1
    return bin_array

def get_all_font_vectors(font_data):
    return np.array([to_bin_array(c).flatten() for c in font_data])


def plot_font_pair(original, reconstructed):
    cmap = plt.get_cmap('binary')

    original_character_template = original.reshape(7, 5)
    reconstructed_character_template = reconstructed.reshape(7, 5)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    sns.heatmap(original_character_template, ax=axes[0], cbar=False, square=True, cmap=cmap, linecolor='k', linewidth=0.2)
    axes[0].set_title(f"Original")
    sns.heatmap(reconstructed_character_template, ax=axes[1], cbar=False, square=True, cmap=cmap, linecolor='k', linewidth=0.2)
    axes[1].set_title(f"Reconstrucción")
    plt.tight_layout()
    plt.show()