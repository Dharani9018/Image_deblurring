import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import image_to_base64


def run(blur_matrix, blurred_matrix, original_matrix):
    K = np.array(blur_matrix)
    B = np.array(blurred_matrix).astype(np.float64)
    A = np.array(original_matrix).astype(np.float64)

    lam = 0.01
    KtK = K.T @ K
    K_reg_inv = np.linalg.solve(KtK + lam * np.eye(KtK.shape[0]), K.T)

    # recover: solve row-wise and column-wise
    recovered = K_reg_inv @ B @ K_reg_inv.T
    recovered = np.clip(recovered, 0, 255)

    try:
        psnr_score = round(float(psnr(
            A.astype(np.uint8), recovered.astype(np.uint8), data_range=255)), 2)
    except Exception:
        psnr_score = 0.0

    return {
        "recovered_image": image_to_base64(recovered),
        "psnr": psnr_score,
        "x_hat_preview": [round(float(v), 3) for v in recovered[0, :6].tolist()]
    }
