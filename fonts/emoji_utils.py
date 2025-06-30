import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from typing import Tuple, List, Optional

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
    
    # Si el emoji está aplanado (1024 elementos), redimensionarlo a (16, 16, 4)
    if emoji.ndim == 1:
        emoji_img = emoji.reshape(16, 16, 4)
    else:
        emoji_img = emoji
    
    # Asegurarse de que los valores estén en [0, 1]
    emoji_img = np.clip(emoji_img, 0, 1)
    
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))  # Figura cuadrada para 16x16
    
    # Usar imshow en lugar de heatmap para mostrar imágenes RGBA
    axes.imshow(emoji_img)
    axes.set_title("Emoji Generado")
    axes.axis('off')  # Ocultar ejes
    
    plt.tight_layout()
    
    # Asegurar que el directorio existe
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", f"{file_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Emoji guardado en: {output_path}")

def plot_font_grid_emoji(originals, outputs, pairs_per_row=5, text=""):
    
    num_pairs = len(originals)
    num_rows = int(np.ceil(num_pairs / pairs_per_row))

    fig, axes = plt.subplots(
        num_rows, pairs_per_row * 2,
        figsize=(pairs_per_row * 4, num_rows * 4)  # Más grande para 16x16
    )

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (original, reconstructed) in enumerate(zip(originals, outputs)):
        row = idx // pairs_per_row
        col_base = (idx % pairs_per_row) * 2

        # Si los datos están aplanados, los redimensionamos a (16, 16, 4)
        if original.ndim == 1:
            original_img = original.reshape(16, 16, 4)
            reconstructed_img = reconstructed.reshape(16, 16, 4)
        else:
            original_img = original
            reconstructed_img = reconstructed

        # Asegurarse de que los valores estén en [0, 1]
        original_img = np.clip(original_img, 0, 1)
        reconstructed_img = np.clip(reconstructed_img, 0, 1)

        # Original
        axes[row][col_base].imshow(original_img)
        axes[row][col_base].set_title("Original")
        axes[row][col_base].axis('off')

        # Reconstrucción
        axes[row][col_base + 1].imshow(reconstructed_img)
        axes[row][col_base + 1].set_title("Reconstrucción")
        axes[row][col_base + 1].axis('off')

    # Ocultar axes vacíos si los hay
    for idx in range(num_pairs, num_rows * pairs_per_row):
        row = idx // pairs_per_row
        col_base = (idx % pairs_per_row) * 2
        if row < num_rows and col_base < pairs_per_row * 2:
            axes[row][col_base].axis('off')
            axes[row][col_base + 1].axis('off')

    # Asegurar que el directorio existe
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", f"emoji_grid_rgba_{text}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Grid guardado en: {output_path}")

def process_folder(
    input_folder: str,
    output_size: Tuple[int, int] = (16, 16),
    flatten: bool = True
) -> Tuple[np.ndarray, Tuple[int, int]]:
    image_arrays: List[np.ndarray] = []
    shape = output_size

    files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".png")
    ])

    for f in files:
        img_path = os.path.join(input_folder, f)
        img = Image.open(img_path).convert("RGBA").resize(shape, Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # Normalizado a [0, 1]

        if flatten:
            arr = arr.reshape(-1)  # Vector de 1024 (si es 16x16x4)
        image_arrays.append(arr)

    X = np.stack(image_arrays)
    return X, shape

def rgba_array_to_png(
    arr: np.ndarray,
    path: str,
    original_shape: Optional[Tuple[int, int]] = None
) -> None:
    if arr.ndim == 1:
        if original_shape is None:
            raise ValueError("Flat array provided — supply original_shape=(H, W)")
        H, W = original_shape
        arr = arr.reshape((H, W, 4))  # RGBA

    arr = np.clip(arr, 0.0, 1.0)  # Asegura que esté en [0, 1]
    arr = (arr * 255).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGBA")
    img.save(path)
