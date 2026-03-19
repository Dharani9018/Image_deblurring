import numpy as np
import matplotlib.pyplot as plt
from utils import plot_to_base64


def run(blur_matrix):
    K = np.array(blur_matrix)

    # eigenvalues of the covariance matrix KᵀK
    KtK = K.T @ K
    eigenvalues = np.linalg.eigvalsh(KtK)
    eigenvalues_sorted = sorted(eigenvalues.tolist(), reverse=True)

    # plot
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(eigenvalues_sorted[:12])),
           eigenvalues_sorted[:12], color="#7F77DD")
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    ax.set_title("eigenvalue spectrum of KᵀK")
    fig.tight_layout()

    return {
        "eigenvalues_preview": [round(v, 4) for v in eigenvalues_sorted[:8]],
        "dominant_eigenvalue": round(eigenvalues_sorted[0], 4),
        "spectrum_plot": plot_to_base64(fig)
    }
