import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def to_bin_array_emoji(encoded_caracter):
    bin_array = np.zeros((8, 9), dtype=int)
    for row in range(8):
        current_row = encoded_caracter[row]
        for col in range(0, 9):
            bin_array[row][8-col] = current_row & 1
            current_row >>= 1
    return bin_array

def get__all_font_vectors_emoji(font_data):
    return np.array([to_bin_array_emoji(c).flatten() for c in font_data])

def plot_font_single_emoji(emoji, file_name="generated_emoji.png"):

    cmap = plt.get_cmap('binary')

    original_emoji_template = emoji.reshape(8, 9)

    fig, axes = plt.subplots(1, 1, figsize=(6, 3))
    sns.heatmap(original_emoji_template, ax=axes, cbar=False, square=True, cmap=cmap, linecolor='k', linewidth=0, xticklabels=False, yticklabels=False)
    plt.tight_layout()

    output_path = os.path.join("results", f"{file_name}.png")
    plt.savefig(output_path)
    plt.close(fig)

def plot_font_grid_emoji(originals, outputs, pairs_per_row=5):

    cmap = plt.get_cmap('binary')

    num_pairs = len(originals)
    num_rows = int(np.ceil(num_pairs / pairs_per_row))

    fig, axes = plt.subplots(
        num_rows, pairs_per_row * 2,
        figsize=(pairs_per_row * 2, num_rows * 2)
    )

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (original, reconstructed) in enumerate(zip(originals, outputs)):
        row = idx // pairs_per_row
        col_base = (idx % pairs_per_row) * 2

        original_template = original.reshape(8, 9)
        reconstructed_template = reconstructed.reshape(8, 9)

        # Original
        sns.heatmap(
            original_template,
            ax=axes[row][col_base],
            cbar=False,
            square=True,
            cmap=cmap,
            linecolor='k',
            linewidth=0,
            xticklabels=False,
            yticklabels=False
        )

        # Reconstrucción
        sns.heatmap(
            reconstructed_template,
            ax=axes[row][col_base + 1],
            cbar=False,
            square=True,
            cmap=cmap,
            linecolor='k',
            linewidth=0,
            xticklabels=False,
            yticklabels=False
        )

    plt.tight_layout()
    output_path = os.path.join("results", f"emoji_grid.png")
    plt.savefig(output_path)
    plt.close(fig)