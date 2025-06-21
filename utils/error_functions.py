from typing import List, Union, Callable
import numpy as np
ErrorFunctionType = Callable[[List[float]], float]

def squared_error(errors:List[float])->Union[int, float]:
    return 0.5 * np.sum(errors ** 2)

def squared_error_adjusted(errors:List[float])->Union[int, float]:
    return (0.5 * np.sum(errors ** 2)) / len(errors)

def mean_error(errors:List[float])->float:
    return np.mean(np.abs(errors)) 
