import numpy as np

def array_top_n_indexes(array, n):
    return np.array(array).argsort()[::-1][:n]
