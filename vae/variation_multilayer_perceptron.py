import numpy as np
from typing import List
from utils.activation_functions import ActivationFunctionType

class MultiLayerPerceptron:
    def __init__(self, input_vector_size:int, activation_function:ActivationFunctionType):
        self.input_vector_size = input_vector_size
        self.activation_function = activation_function

    def predict(self, input_set:List[any], weights:List[float], beta:float = 1.0):
            h_supra_mu = np.dot(weights, input_set) 
            return self.activation_function(h_supra_mu, beta), h_supra_mu
    
class LatentSpaceMultiLayerPerceptron(MultiLayerPerceptron):
    def __init__(self, input_vector_size:int, activation_function:ActivationFunctionType):
        super().__init__(input_vector_size, activation_function)

    def predict(self, media:float, variance:float):
        epsilon = np.random.randn()  # Sample from a standard normal distribution
        z = media + np.sqrt(variance) * epsilon  # Reparameterization trick
        h_supra_mu = np.array([media, variance, z])
        return self.activation_function(h_supra_mu, 1.0), h_supra_mu
    