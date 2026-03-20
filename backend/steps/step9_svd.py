import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils import image_to_base64


def truncated_svd_reconstruct(K, B, k):
    U, S, Vt = np.linalg.svd(K, full_matrices=False)
    S_trunc = np.zeros_like(S)
    S_trunc[:k] = S[:k]
    K_approx = U @ np.diag(S_trunc) @ Vt

    recovered = B.copy().astype(np.float64)
    for col in range(B.shape[1]):
        x_hat, _, _, _ = np.linalg.lstsq(K_approx, B[:24, col], rcond=None)
        recovered[:24, col] = x_hat

    return np.clip(recovered, 0, 255)


def run(blur_matrix, blurred_image, original_image):
    K = np.array(blur_matrix)
    B = np.array(blurred_image).astype(np.float64)
    A = np.array(original_image).astype(np.float64)

    orig_u8 = np.clip(A, 0, 255).astype(np.uint8)

    _, S_full, _ = np.linalg.svd(K, full_matrices=False)
    max_k = len(S_full)

    results = {}
    for k in [5, min(20, max_k), min(max_k, 24)]:
        recovered = truncated_svd_reconstruct(K, B, k)
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
