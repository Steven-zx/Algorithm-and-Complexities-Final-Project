import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_features(matrix):
    """Normalizes the feature matrix to a range of 0 to 1."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(matrix)

def normalize_matrix_qr(matrix):
    """Normalize input matrix using QR Decomposition (using the Q factor)."""
    q, r = np.linalg.qr(matrix)
    return q

def gaussian_elimination(A, b):
    """Solves Ax = b using Gaussian Elimination."""
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    # Augmented matrix [A|b]
    Ab = np.concatenate((A, b.reshape(-1, 1)), axis=1)

    # Forward elimination
    for i in range(n):
        pivot = i
        for j in range(i + 1, n):
            if abs(Ab[j][i]) > abs(Ab[pivot][i]):
                pivot = j
        Ab[[i, pivot]] = Ab[[pivot, i]]

        if Ab[i][i] == 0:
            continue

        for j in range(i + 1, n):
            factor = Ab[j][i] / Ab[i][i]
            Ab[j] = Ab[j] - factor * Ab[i]

    # Back-substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ax = np.sum(Ab[i][i + 1:n] * x[i + 1:])
        x[i] = (Ab[i][n] - sum_ax) / Ab[i][i]

    return x

def cramer_rule(A, b):
    """Solves Ax = b using Cramer's Rule."""
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
    Attempts to rank internships using Gaussian Elimination on normalized data,
    handling "Not Important" weights.
    """
    normalized_internships = normalize_features(internships)
    num_internships = normalized_internships.shape[0]
    num_features = normalized_internships.shape[1]

    scores = []
    for i in range(num_internships):
        internship_features = normalized_internships[i, :]
        score = 0.0
        for j in range(num_features):
            score += internship_features[j] * user_weights[j]

        scores.append(score)

    return np.array(scores)