import numpy as np
from PIL import Image
import io
from utils import image_to_base64, matrix_preview


def run(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((64, 64))

    color_array = np.array(img).astype(np.float64)      # 64×64×3 for display
    gray = np.mean(color_array, axis=2)                  # 64×64 for math

    return {
        "original_image": image_to_base64(color_array.astype(np.uint8)),  # color
        "matrix_preview": matrix_preview(gray),
        "shape": list(gray.shape),
        "matrix": gray.tolist()                          # grayscale for pipeline
    }
