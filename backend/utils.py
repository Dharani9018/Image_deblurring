import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def image_to_base64(image_array):
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="PNG", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def matrix_preview(matrix, size=4):
    preview = matrix[:size, :size].tolist()
    return [[round(v, 2) for v in row] for row in preview]
