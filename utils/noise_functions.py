import numpy as np
import random

def gaussian_noise(data, std):
    """
    Agrega ruido gaussiano con desviación estándar 'std'.
    """
    noise = np.random.normal(0, std, np.shape(data))
    noisy_data = data + noise
    return np.clip(noisy_data, 0, 1)

def salt_and_pepper_noise(data, noise_threshold):
    """
    Invierte bits aleatoriamente con una probabilidad 'noise_threshold'.
    """
    noisy_data = np.copy(data)
    for i in range(len(noisy_data)):
        if random.random() < noise_threshold:
            noisy_data[i] = 1 - noisy_data[i]  # Flip bit
    return noisy_data
