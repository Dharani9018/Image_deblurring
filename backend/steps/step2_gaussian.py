import numpy as np
from scipy.linalg import lu
from utils import image_to_base64, matrix_preview


def build_gaussian_kernel(size, sigma=2.0):
    k = np.arange(size)
    center = size // 2
    g = np.exp(-0.5 * ((k - center) / sigma) ** 2)
    g = g / g.sum()
    K = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            K[i, j] = g[abs(i - j) % size]
    return K


def numpy_rref_preview(matrix, size=4):
    M = matrix[:size, :size].astype(float).copy()
    rows, cols = M.shape
    for col in range(cols):
        pivot = None
        for row in range(col, rows):
            if abs(M[row, col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        M[[col, pivot]] = M[[pivot, col]]
        M[col] = M[col] / M[col, col]
        for row in range(rows):
            if row != col:
                M[row] -= M[row, col] * M[col]
    return [[round(float(v), 2) for v in row] for row in M]


def run(image_matrix):
    A = np.array(image_matrix).astype(np.float64)
    size = A.shape[0]

    K = build_gaussian_kernel(size, sigma=2.0)

    blurred = np.clip(K @ A @ K.T, 0, 255)

    rref_preview = numpy_rref_preview(K, size=4)

    K_small = K[:8, :8]
    P, L, U = lu(K_small)

    rank_full = int(np.linalg.matrix_rank(K))
    nullity = size - rank_full

    pivots = [i for i in range(K_small.shape[0]) if abs(U[i, i]) > 1e-10]

    return {
        "blurred_image": image_to_base64(blurred),
        "blurred_matrix": blurred.tolist(),
        "blur_matrix": K.tolist(),
        "rref_preview": rref_preview,
        "pivots": pivots,
        "rank": rank_full,
        "nullity": nullity,
        "L_preview": matrix_preview(L),
        "U_preview": matrix_preview(U)
    }
