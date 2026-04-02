import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def image_to_base64(image_array):
    arr = np.array(image_array)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3:
        img = Image.fromarray(arr, mode="RGB")
    else:
        img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="PNG", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def matrix_preview(matrix, size=4):
    preview = np.array(matrix)[:size, :size].tolist()
    return [[round(float(v), 2) for v in row] for row in preview]
