import numpy as np
from utils import image_to_base64


def run(blur_matrix, blurred_matrix):
    K = np.array(blur_matrix)
    B = np.array(blurred_matrix).astype(np.float64)

    KtK = K.T @ K
    KtK_inv = np.linalg.pinv(KtK)
    P = K @ KtK_inv @ K.T

    projected = np.clip(P @ B, 0, 255)

    return {
        "projected_image": image_to_base64(projected),
        "projection_matrix_preview": [[round(float(P[i][j]), 4) for j in range(4)]
                                      for i in range(4)],
        "is_idempotent": bool(np.allclose(P @ P, P, atol=1e-4))
    }
