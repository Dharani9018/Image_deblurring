import numpy as np


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v.copy().astype(float)
        for b in basis:
            w -= np.dot(w, b) * b  
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            basis.append(w / norm)
    return np.array(basis)


def run(blur_matrix):
    K = np.array(blur_matrix)
    rank = int(np.linalg.matrix_rank(K))

    vectors = [K[:, i] for i in range(min(rank, 10))]
    ortho_basis = gram_schmidt(vectors)

    dot_check = float(np.dot(ortho_basis[0], ortho_basis[1])) if len(ortho_basis) > 1 else 0.0

    return {
        "num_vectors_in": rank,
        "num_vectors_out": len(ortho_basis),
        "orthogonal_basis_preview": [[round(float(v), 4) for v in row[:4]]
                                     for row in ortho_basis[:3]],
        "dot_product_check": round(dot_check, 8),
        "is_orthogonal": abs(dot_check) < 1e-6
    }
