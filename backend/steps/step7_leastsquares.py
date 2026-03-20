import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import image_to_base64


def run(blur_matrix, blurred_image, original_image):
    K = np.array(blur_matrix)
    B = np.array(blurred_image).astype(np.float64)
    A = np.array(original_image).astype(np.float64)

    recovered_full = B.copy()

    for col in range(B.shape[1]):
        x_hat, _, _, _ = np.linalg.lstsq(K, B[:24, col], rcond=None)
        recovered_full[:24, col] = x_hat

    recovered_full = np.clip(recovered_full, 0, 255)

    orig_u8 = np.clip(A, 0, 255).astype(np.uint8)
    rec_u8 = recovered_full.astype(np.uint8)

    try:
        psnr_score = round(float(psnr(orig_u8, rec_u8, data_range=255)), 2)
    except Exception:
        psnr_score = 0.0

    return {
        "recovered_image": image_to_base64(recovered_full),
        "psnr": psnr_score,
        "x_hat_preview": [round(float(v), 3) for v in recovered_full[0, :6].tolist()]
    }
