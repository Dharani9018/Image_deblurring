import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from utils import image_to_base64


def truncated_svd_reconstruct(K, blurred_patch, k):
    U, S, Vt = np.linalg.svd(K, full_matrices=False)
    S_trunc = np.zeros_like(S)
    S_trunc[:k] = S[:k]
    K_approx = U @ np.diag(S_trunc) @ Vt

    b = blurred_patch.flatten()
    x_hat, _, _, _ = np.linalg.lstsq(K_approx, b, rcond=None)
    return x_hat.reshape(blurred_patch.shape)


def run(blur_matrix, blurred_image, original_image):
    K = np.array(blur_matrix)
    B = np.array(blurred_image)
    A = np.array(original_image)

    patch_b = B[:24, :24]
    patch_a = A[:24, :24]

    results = {}
    for k in [5, 20, min(50, K.shape[0])]:
        recovered_patch = truncated_svd_reconstruct(K, patch_b, k)
        recovered_full = B.copy()
        recovered_full[:24, :24] = recovered_patch
        recovered_full = np.clip(recovered_full, 0, 255)

        orig_u8 = np.clip(A, 0, 255).astype(np.uint8)
        rec_u8 = recovered_full.astype(np.uint8)

        psnr_score = round(float(psnr(orig_u8, rec_u8)), 2)
        ssim_score = round(float(ssim(orig_u8, rec_u8, data_range=255)), 3)

        results[f"k{k}"] = {
            "image": image_to_base64(recovered_full),
            "psnr": psnr_score,
            "ssim": ssim_score
        }

    _, S, _ = np.linalg.svd(K, full_matrices=False)

    return {
        "reconstructions": results,
        "singular_values": [round(float(s), 4) for s in S[:10]]
    }
