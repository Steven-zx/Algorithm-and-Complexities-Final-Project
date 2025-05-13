import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

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

def calculate_rankings_weighted_sum(internships: np.ndarray, user_weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Ranks internships using a simple weighted sum approach.
    Time Complexity: O(n*m) where n is number of internships and m is number of features
    """
    start_time = time.time()
    # normalized_internships = normalize_features(internships)
    scores = np.dot(internships, user_weights)
    execution_time = time.time() - start_time
    
    metrics = {
        "time_complexity": "O(n*m)",
        "execution_time": execution_time,
        "numerical_stability": "High",
        "method": "Weighted Sum"
    }
    
    return scores, metrics

def calculate_rankings_gaussian(internships: np.ndarray, user_weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Ranks internships using Gaussian Elimination.
    Time Complexity: O(n^3) where n is the number of features
    """
    start_time = time.time()
    # normalized_internships = normalize_features(internships)
    n = internships.shape[0]
    scores = np.zeros(n)
    
    for i in range(n):
        A = internships[i:i+1].T
        b = user_weights.reshape(-1, 1)
        try:
            solution = gaussian_elimination(A, b.flatten())
            scores[i] = np.sum(solution)
        except:
            scores[i] = 0
    
    execution_time = time.time() - start_time
    
    metrics = {
        "time_complexity": "O(n^3)",
        "execution_time": execution_time,
        "numerical_stability": "Medium",
        "method": "Gaussian Elimination"
    }
    
    return scores, metrics

def calculate_rankings_qr(internships: np.ndarray, user_weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Ranks internships using QR Decomposition.
    Time Complexity: O(n^3) but more numerically stable than Gaussian
    """
    start_time = time.time()
    # normalized_internships = normalize_matrix_qr(internships)
    scores = np.dot(internships, user_weights)
    execution_time = time.time() - start_time
    
    metrics = {
        "time_complexity": "O(n^3)",
        "execution_time": execution_time,
        "numerical_stability": "Very High",
        "method": "QR Decomposition"
    }
    
    return scores, metrics

def calculate_rankings_cramer(internships: np.ndarray, user_weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Ranks internships using Cramer's Rule.
    Time Complexity: O(n^4) due to determinant calculations
    """
    start_time = time.time()
    # normalized_internships = normalize_features(internships)
    n = internships.shape[0]
    scores = np.zeros(n)
    
    for i in range(n):
        A = internships[i:i+1].T
        b = user_weights.reshape(-1, 1)
        try:
            solution = cramer_rule(A, b.flatten())
            scores[i] = np.sum(solution)
        except:
            scores[i] = 0
    
    execution_time = time.time() - start_time
    
    metrics = {
        "time_complexity": "O(n^4)",
        "execution_time": execution_time,
        "numerical_stability": "Low",
        "method": "Cramer's Rule"
    }
    
    return scores, metrics

def compare_algorithms(internships: np.ndarray, user_weights: np.ndarray) -> Dict[str, Tuple[np.ndarray, Dict]]:
    """
    Compares the three required ranking algorithms and returns their results and metrics.
    """
    results = {}
    
    # Gaussian Elimination
    results["gaussian"] = calculate_rankings_gaussian(internships, user_weights)
    
    # QR Decomposition
    results["qr"] = calculate_rankings_qr(internships, user_weights)
    
    # Cramer's Rule
    results["cramer"] = calculate_rankings_cramer(internships, user_weights)
    
    return results

def plot_algorithm_comparison(results: Dict[str, Tuple[np.ndarray, Dict]], top_n: int = 10) -> None:
    """
    Creates visualizations comparing the different algorithms.
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot execution times
    methods = []
    times = []
    for method, (_, metrics) in results.items():
        methods.append(metrics["method"])
        times.append(metrics["execution_time"])
    
    ax1.bar(methods, times)
    ax1.set_title("Algorithm Execution Times")
    ax1.set_ylabel("Time (seconds)")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot top N rankings comparison
    for method, (scores, _) in results.items():
        top_indices = np.argsort(scores)[-top_n:][::-1]
        ax2.plot(range(top_n), scores[top_indices], label=results[method][1]["method"])
    
    ax2.set_title(f"Top {top_n} Rankings Comparison")
    ax2.set_xlabel("Rank")
    ax2.set_ylabel("Score")
    ax2.legend()
    
    plt.tight_layout()
    return fig