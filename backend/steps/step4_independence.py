import numpy as np


def run(blur_matrix):
    K = np.array(blur_matrix)

    Q, R, pivot = np.linalg.svd(K, full_matrices=False), None, None
    _, R_qr, pivot_qr = np.linalg.svd(K), None, None

    Q_qr, R_qr = np.linalg.qr(K)
    rank = int(np.linalg.matrix_rank(K))
    diag_R = np.abs(np.diag(R_qr))

    threshold = diag_R.max() * 1e-10
    independent_cols = [int(i) for i in range(len(diag_R)) if diag_R[i] > threshold][:rank]

    singular_values = np.linalg.svd(K, compute_uv=False)

    return {
        "rank": rank,
        "independent_columns": independent_cols[:8],
        "basis_preview": [[round(float(K[i][j]), 3) for j in range(min(4, K.shape[1]))]
                          for i in independent_cols[:4]],
        "singular_values_preview": [round(float(s), 4) for s in singular_values[:6]]
    }
