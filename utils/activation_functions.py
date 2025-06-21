from typing import Callable
import numpy as np

ActivationFunctionType = Callable[[float, float], float]

def tanh(x:float, beta:float)->float:
    return np.tanh(beta * x)

def prime_tanh(x:float, beta:float)->float:
    return beta * (1 - (tanh(x,beta) ** 2))

def logistic(x:float, beta:float)->float:
    return 1 / (1 + np.exp(-2 * beta * x))

def prime_logistic(x:float, beta:float)->float:
    return 2 * beta * logistic(x, beta) * (1 - logistic(x, beta))

def relu(x, beta:float=1.0):
    return np.maximum(0, x)

def relu_derivative(x, beta:float=1.0):
    result = np.zeros_like(x, dtype=float)
    result[x > 0] = 1.0
    
    return result