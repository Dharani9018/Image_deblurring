import numpy as np
import matplotlib.pyplot as plt
from utils import plot_to_base64


def run(blur_matrix):
    K = np.array(blur_matrix)

    KtK = K.T @ K
    eigenvalues = np.linalg.eigvalsh(KtK)
    eigenvalues_sorted = sorted(eigenvalues.tolist(), reverse=True)

    singular_values = np.linalg.svd(K, compute_uv=False)

    # log scale plot — shows decay clearly
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    ax1.bar(range(len(eigenvalues_sorted[:12])),
            eigenvalues_sorted[:12], color="#7F77DD")
    ax1.set_title("eigenvalue spectrum")
    ax1.set_xlabel("index")
    ax1.set_ylabel("eigenvalue")

    ax2.semilogy(singular_values[:20], color="#1D9E75", marker="o", markersize=3)
    ax2.set_title("singular value decay (log)")
    ax2.set_xlabel("index")
    ax2.set_ylabel("singular value")

    fig.tight_layout()

    return {
        "eigenvalues_preview": [round(v, 4) for v in eigenvalues_sorted[:8]],
        "dominant_eigenvalue": round(eigenvalues_sorted[0], 4),
        "singular_values": [round(float(s), 4) for s in singular_values[:8]],
        "spectrum_plot": plot_to_base64(fig)
    }
