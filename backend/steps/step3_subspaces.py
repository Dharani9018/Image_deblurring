import numpy as np


def run(blur_matrix):
    K = np.array(blur_matrix)

    rank = int(np.linalg.matrix_rank(K))
    nullity = K.shape[1] - rank

    # column space — linearly independent columns
    _, _, Vt = np.linalg.svd(K)
    col_space_basis = Vt[:rank].tolist()

    # null space via SVD (rows of Vt corresponding to near-zero singular values)
    null_space_basis = Vt[rank:].tolist()

    return {
        "rank": rank,
        "nullity": nullity,
        "col_space_basis_preview": [row[:4] for row in col_space_basis[:3]],
        "null_space_basis_preview": [row[:4] for row in null_space_basis[:3]],
        "theorem": f"rank({rank}) + nullity({nullity}) = {rank + nullity} = n"
    }
