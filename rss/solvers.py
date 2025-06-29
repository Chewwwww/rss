import numpy as np

#from numpy import linalg as LA

def rk(A, x, b):
    m = A.shape[0]

    rng = np.random.default_rng()

    ind = rng.integers(low=0, high=m)

    x = x + ((b[ind] - np.dot(A[ind, :], x)) / (np.linalg.norm(A[ind, :]) ** 2)) * A[ind, :]

    return x
