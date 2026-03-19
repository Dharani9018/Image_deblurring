import numpy as np
from utils import image_to_base64


def run(blur_matrix, blurred_image):
    K = np.array(blur_matrix)
    B = np.array(blurred_image)

    patch = B[:24, :24]
    b = patch.flatten()

    # projection onto column space of K: P = K(KᵀK)⁻¹Kᵀ
    KtK = K.T @ K
    KtK_inv = np.linalg.pinv(KtK)
    P = K @ KtK_inv @ K.T

    projected = P @ b
    projected_patch = projected.reshape(24, 24)

    projected_full = np.zeros_like(B)
    projected_full[:24, :24] = projected_patch

    return {
        "projected_image": image_to_base64(projected_full),
        "projection_matrix_preview": [[round(float(P[i][j]), 4) for j in range(4)]
                                      for i in range(4)],
        "is_idempotent": bool(np.allclose(P @ P, P, atol=1e-6))
    }
