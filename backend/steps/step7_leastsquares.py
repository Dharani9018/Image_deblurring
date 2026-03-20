import numpy as np
from utils import image_to_base64


def run(blur_matrix, image_matrix, original_image):
    K = np.array(blur_matrix)
    A = np.array(image_matrix)

    KtK = K.T @ K
    KtK_inv = np.linalg.pinv(KtK)
    P = K @ KtK_inv @ K.T

    projected_full = A.copy()          # start with full image, not zeros

    for col in range(A.shape[1]):
        b_col = A[:24, col]
        projected_full[:24, col] = P @ b_col
        # rows 24–64 stay as original — no black patch

    projected_full = np.clip(projected_full, 0, 255)

    return {
        "projected_image": image_to_base64(projected_full),
        "projection_matrix_preview": [[round(float(P[i][j]), 4) for j in range(4)]
                                      for i in range(4)],
        "is_idempotent": bool(np.allclose(P @ P, P, atol=1e-6))
    }
