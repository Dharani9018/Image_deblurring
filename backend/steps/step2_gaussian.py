import numpy as np
from scipy.linalg import lu
from sympy import Matrix
from utils import image_to_base64, matrix_preview
from scipy.ndimage import uniform_filter


def build_blur_kernel(size=24):
    K = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if abs(i - j) <= 2:
                K[i][j] = 1.0 / (2 * 2 + 1)
    return K


def run(image_matrix):
    A = np.array(image_matrix)

    # blur the FULL image using a simple uniform filter so it looks visually blurred
    blurred_full = uniform_filter(A, size=9).astype(np.float64)

    # still use the 24×24 patch for the linear algebra parts
    patch = A[:24, :24]
    K = build_blur_kernel(24)
    blurred_patch = K @ patch

    # RREF using sympy
    sym_K = Matrix(K.tolist())
    rref_matrix, pivots = sym_K.rref()
    rref_preview = [[float(rref_matrix[i, j]) for j in range(min(4, K.shape[1]))]
                    for i in range(min(4, K.shape[0]))]

    # LU decomposition
    P, L, U = lu(K)

    rank = len(pivots)
    nullity = K.shape[1] - rank

    return {
        "blurred_image": image_to_base64(blurred_full),
        "blur_matrix": K.tolist(),
        "rref_preview": rref_preview,
        "pivots": list(pivots),
        "rank": rank,
        "nullity": nullity,
        "L_preview": matrix_preview(L),
        "U_preview": matrix_preview(U)
    }
