import numpy as np
import jpeglib


def dct_matrix(n, m):
    """
    Return a DCT matrix M of size (n*m, n*m) such that np.dot(M, x) is the DCT transform of x and np.dot(M.T,x) is the
    inverse DCT transform of x.
    """
    matrix = np.zeros((n * m, n * m))
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for l in range(m):
                    matrix[i * m + j, k * m + l] = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.cos(
                        np.pi * (2 * j + 1) * l / (2 * m))

    return (matrix / np.linalg.norm(matrix, axis=0)).T


def round(x):
    return np.multiply(np.sign(x), np.trunc(np.abs(x) + 1 / 2))


def read_jpeg(path, return_dct=False):
    if return_dct:
        return jpeglib.read_dct(path).Y  # Only luminance
    else:
        return jpeglib.read_spatial(path).spatial
