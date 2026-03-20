# step6_projection.py
import numpy as np
from utils import image_to_base64


def run(blur_matrix, image_matrix):
    K = np.array(blur_matrix)
    A = np.array(image_matrix)

    patch = A[:24, :24]
    b = patch.flatten()[:24]          # ← take only first 24 elements to match K's rows

    KtK = K.T @ K
    KtK_inv = np.linalg.pinv(KtK)
    P = K @ KtK_inv @ K.T             # P is 24×24

    projected = P @ b                 # both 24 now, no mismatch
    projected_patch = np.outer(projected, np.ones(24))  # reshape back to 24×24

    projected_full = np.zeros_like(A)
    projected_full[:24, :24] = projected_patch

    return {
        "projected_image": image_to_base64(projected_full),
        "projection_matrix_preview": [[round(float(P[i][j]), 4) for j in range(4)]
                                      for i in range(4)],
        "is_idempotent": bool(np.allclose(P @ P, P, atol=1e-6))
    }
