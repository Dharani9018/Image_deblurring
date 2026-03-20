import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import image_to_base64


def run(blur_matrix, blurred_image, original_image):
    K = np.array(blur_matrix)           # 24×24
    B = np.array(blurred_image)
    A = np.array(original_image)

    results = []
    recovered_full = B.copy()

    # process column by column — K is 24×24, so solve one column at a time
    for col in range(B.shape[1]):
        b_col = B[:24, col] if col < B.shape[1] else B[:24, 0]
        x_hat, _, _, _ = np.linalg.lstsq(K, b_col, rcond=None)
        recovered_full[:24, col] = x_hat

    recovered_full = np.clip(recovered_full, 0, 255)
    orig_u8 = np.clip(A, 0, 255).astype(np.uint8)
    rec_u8 = recovered_full.astype(np.uint8)

    psnr_score = round(float(psnr(orig_u8, rec_u8)), 2)

    return {
        "recovered_image": image_to_base64(recovered_full),
        "psnr": psnr_score,
        "x_hat_preview": [round(float(v), 3) for v in recovered_full[0, :6].tolist()]
    }
