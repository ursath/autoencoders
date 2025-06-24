import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

def plot_font_single(charcacter, file_name="generated_character.png"):

    cmap = plt.get_cmap('binary')

    original_character_template = charcacter.reshape(7, 5)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    sns.heatmap(original_character_template, ax=axes[0], cbar=False, square=True, cmap=cmap, linecolor='k', linewidth=0, xticklabels=False, yticklabels=False)
    axes[0].set_title(f"Original")
    plt.tight_layout()

    output_path = os.path.join("results", f"{file_name}.png")
    plt.savefig(output_path)
    plt.close(fig) 

def plot_font_pair(original, reconstructed, character):

    folder_path = "results/characters"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cmap = plt.get_cmap('binary')

    original_character_template = original.reshape(7, 5)
    reconstructed_character_template = reconstructed.reshape(7, 5)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    sns.heatmap(original_character_template, ax=axes[0], cbar=False, square=True, cmap=cmap, linecolor='k', linewidth=0, xticklabels=False, yticklabels=False)
    axes[0].set_title(f"Original")
    sns.heatmap(reconstructed_character_template, ax=axes[1], cbar=False, square=True, cmap=cmap, linecolor='k', linewidth=0, xticklabels=False, yticklabels=False)
    axes[1].set_title(f"Reconstrucción")
    plt.tight_layout()

    output_path = os.path.join("results/characters", f"{character}.png")
    plt.savefig(output_path)
    plt.close(fig) 



def plot_font_grid(originals, outputs, pairs_per_row=5):

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

        original_template = original.reshape(7, 5)
        reconstructed_template = reconstructed.reshape(7, 5)

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
    output_path = os.path.join("results", f"grid.png")
    plt.savefig(output_path)
    plt.close(fig)


def pixel_array_to_char(vector, font_data, first_char=0x60):
    all_vectors = get_all_font_vectors(font_data)
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']

    for i, font_vector in enumerate(all_vectors):
        if np.array_equal(vector, font_vector):
            character = chr(first_char + i)
            if character in invalid_chars:
                return f"char_{ord(character)}"
            else:
                return character
    return None