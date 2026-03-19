import numpy as np
from skimage import color
from PIL import Image
import io
from utils import image_to_base64, matrix_preview


def run(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((64, 64))
    gray = color.rgb2gray(np.array(img)) * 255

    return {
        "original_image": image_to_base64(gray),
        "matrix_preview": matrix_preview(gray),
        "shape": list(gray.shape),
        "matrix": gray.tolist()
    }
