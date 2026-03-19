import numpy as np


def run(blur_matrix):
    K = np.array(blur_matrix)

    # find linearly independent columns using QR with pivoting
    Q, R, pivot = np.linalg.qr(K, mode='complete'), None, None
    _, R, pivot = np.linalg.svd(K), None, None

    # simpler: use rank to find independent columns
    rank = int(np.linalg.matrix_rank(K))
    singular_values = np.linalg.svd(K, compute_uv=False)

    # columns with non-negligible contribution
    independent_cols = list(range(rank))

    basis = K[:, independent_cols].tolist()

    return {
        "rank": rank,
        "independent_columns": independent_cols,
        "basis_preview": [[round(basis[i][j], 3) for j in range(min(4, len(basis[0])))]
                          for i in range(min(4, len(basis)))],
        "singular_values_preview": [round(float(s), 4) for s in singular_values[:6]]
    }
