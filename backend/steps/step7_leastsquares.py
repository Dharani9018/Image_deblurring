import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import image_to_base64


def run(blur_matrix, blurred_image, original_image):
    K = np.array(blur_matrix)
    B = np.array(blurred_image)
    A = np.array(original_image)

    patch_b = B[:24, :24].flatten()

    # least squares: x̂ = (KᵀK)⁻¹Kᵀb
    x_hat, _, _, _ = np.linalg.lstsq(K, patch_b, rcond=None)

    recovered_patch = x_hat.reshape(24, 24)
    recovered_full = B.copy()
    recovered_full[:24, :24] = recovered_patch

    recovered_clipped = np.clip(recovered_full, 0, 255)
    original_clipped = np.clip(A, 0, 255)

    psnr_score = round(float(psnr(original_clipped.astype(np.uint8),
                                   recovered_clipped.astype(np.uint8))), 2)

    return {
        "recovered_image": image_to_base64(recovered_clipped),
        "psnr": psnr_score,
        "x_hat_preview": [round(float(v), 3) for v in x_hat[:6]]
    }
