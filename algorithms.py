import numpy as np

def normalize_matrix_qr(matrix):
    """
    Normalize input matrix using QR Decomposition.
    Ensures features like allowance and reputation are on the same scale.
    """
    q, r = np.linalg.qr(matrix)
    return q  # normalized matrix


def gaussian_elimination(A, b):
    """
    Solves Ax = b using Gaussian Elimination.
    A: Coefficient matrix (internship attributes × weights)
    b: Vector of weights based on user preferences
    Returns: Score vector for ranking
    """
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    # Forward elimination
    for i in range(n):
        for j in range(i+1, n):
            if A[i][i] == 0:
                continue
            ratio = A[j][i] / A[i][i]
            A[j] = A[j] - ratio * A[i]
            b[j] = b[j] - ratio * b[i]

    # Back-substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_ax = np.sum(A[i][i+1:] * x[i+1:])
        x[i] = (b[i] - sum_ax) / A[i][i]
    
    return x


def cramer_rule(A, b):
    """
    Solves Ax = b using Cramer's Rule.
    A: Square matrix of internship feature scores
    b: Preference-weighted outcome
    Returns: Vector of solution scores
    """
    det_A = np.linalg.det(A)
    if det_A == 0:
        raise ValueError("System has no unique solution.")
    
    n = A.shape[1]
    x = np.zeros(n)

    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = np.linalg.det(A_i) / det_A

    return x


def calculate_rankings(internships, user_weights):
    """
    Main function to rank internships based on user weights.
    internships: NumPy matrix of numeric features (e.g., allowance, reputation)
    user_weights: Vector from user input (weights for each criterion)
    Returns: Ranking scores
    """
    # Normalize to avoid scale bias
    normalized = normalize_matrix_qr(internships)

    # Use Gaussian Elimination to get scores
    scores = gaussian_elimination(normalized, user_weights)

    return scores
