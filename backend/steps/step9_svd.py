import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils import image_to_base64


def truncated_svd_recover(K, B, k, lam=0.01):
    U, S, Vt = np.linalg.svd(K, full_matrices=False)
    S_k = S[:k]
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    S_inv = S_k / (S_k ** 2 + lam)
    K_pinv = Vt_k.T @ np.diag(S_inv) @ U_k.T

    recovered = K_pinv @ B @ K_pinv.T
    return np.clip(recovered, 0, 255)


def run(blur_matrix, blurred_matrix, original_matrix):
    K = np.array(blur_matrix)
    B = np.array(blurred_matrix).astype(np.float64)
    A = np.array(original_matrix).astype(np.float64)

    orig_u8 = A.astype(np.uint8)
    _, S_full, _ = np.linalg.svd(K, full_matrices=False)
    max_k = len(S_full)

    results = {}
    for k in [5, min(20, max_k), max_k]:
        recovered = truncated_svd_recover(K, B, k)
        rec_u8 = recovered.astype(np.uint8)
        try:
            psnr_score = round(float(psnr(orig_u8, rec_u8, data_range=255)), 2)
            ssim_score = round(float(ssim(orig_u8, rec_u8, data_range=255)), 3)
        except Exception:
            psnr_score = 0.0
            ssim_score = 0.0
        results[f"k{k}"] = {
            "image": image_to_base64(recovered),
            "psnr": psnr_score,
            "ssim": ssim_score
        }

    return {
        "reconstructions": results,
        "singular_values": [round(float(s), 4) for s in S_full[:10]]
    }
