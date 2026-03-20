import numpy as np
from utils import image_to_base64


def run(blur_matrix, image_matrix):
    K = np.array(blur_matrix)
    A = np.array(image_matrix)

    KtK = K.T @ K
    KtK_inv = np.linalg.pinv(KtK)
    P = K @ KtK_inv @ K.T

    projected_full = A.copy().astype(np.float64)

    for col in range(A.shape[1]):
        projected_full[:24, col] = P @ A[:24, col]

    projected_full = np.clip(projected_full, 0, 255)

    return {
        "projected_image": image_to_base64(projected_full),
        "projection_matrix_preview": [[round(float(P[i][j]), 4) for j in range(4)]
                                      for i in range(4)],
        "is_idempotent": bool(np.allclose(P @ P, P, atol=1e-6))
    }
