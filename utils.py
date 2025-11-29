import numpy as np
from typing import Tuple, Optional, List


def compute_ordering_exchange_2d(data: np.ndarray, item_i: int, item_j: int) -> Optional[float]:
    """
    Compute the ordering exchange angle between two items in 2D.
    Based on Equation 6 in the paper.

    Args:
        item_i, item_j: Indices of two items

    Returns:
        Angle theta where items exchange order (None if one dominates)
    """
    n_items, n_attrs = data.shape
    if n_attrs != 2:
        raise ValueError("Ordering exchange angle only defined for 2D")

    t_i = data[item_i]
    t_j = data[item_j]

    # Check for dominance
    if np.all(t_i >= t_j) and np.any(t_i > t_j):
        return None  # i dominates j
    if np.all(t_j >= t_i) and np.any(t_j > t_i):
        return None  # j dominates i

    # Compute ordering exchange angle
    numerator = t_j[0] - t_i[0]
    denominator = t_i[1] - t_j[1]

    if abs(denominator) < 1e-10:
        return None

    theta = np.arctan(numerator / denominator)

    # Ensure theta is in [0, pi/2]
    if theta < 0:
        theta += np.pi
    if theta > np.pi/2:
        return None

    return theta

def get_ranking(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Get ranking of items based on weights (higher score = better rank).

    Args:
        weights: d-dimensional weight vector

    Returns:
        Array of item indices in ranked order (best to worst)
    """
    scores = data @ weights
    return np.argsort(-scores)  # Descending order

def verify_stability_2d(data: np.ndarray, ranking: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """
    Verify the stability of a given ranking in 2D.
    Algorithm from Section 3.1 of the paper.

    Args:
        ranking: Array of item indices in ranked order

    Returns:
        Tuple of (stability, (theta_min, theta_max))
    """
    n_items, n_attrs = data.shape
    if n_attrs != 2:
        raise ValueError("2D stability verification only works for 2D data")

    theta_min = 0.0
    theta_max = np.pi / 2

    # Check each adjacent pair in the ranking
    for i in range(len(ranking) - 1):
        item_higher = ranking[i]
        item_lower = ranking[i + 1]

        theta_exchange = compute_ordering_exchange_2d(data, item_higher, item_lower)

        if theta_exchange is None:
            # Check if ranking is valid
            dominates = np.all(data[item_higher] >= data[item_lower]) and np.any(data[item_higher] > data[item_lower])
            if not dominates:
                return 0.0, (0.0, 0.0)  # Invalid ranking
            continue

        # Determine which side of the exchange the ranking is valid
        if data[item_higher, 0] < data[item_lower, 0]:
            # Higher ranked item has smaller x1, so valid for theta < theta_exchange
            theta_max = min(theta_max, theta_exchange)
        else:
            # Higher ranked item has larger x1, so valid for theta > theta_exchange
            theta_min = max(theta_min, theta_exchange)

    if theta_min >= theta_max:
        return 0.0, (0.0, 0.0)

    stability = (theta_max - theta_min) / (np.pi / 2)
    return stability, (theta_min, theta_max)

def enumerate_stable_rankings_2d(data: np.ndarray, n_top: int = 10) -> List[Tuple[np.ndarray, float, Tuple[float, float]]]:
    """
    Enumerate the most stable rankings in 2D using ray sweeping.
    Based on Algorithm 1 (RAYSWEEPING) from the paper.

    Args:
        n_top: Number of top stable rankings to return

    Returns:
        List of tuples (ranking, stability, (theta_min, theta_max))
    """
    n_items, n_attrs = data.shape
    if n_attrs != 2:
        raise ValueError("2D enumeration only works for 2D data")

    # Compute all ordering exchanges
    exchanges = [] # Should be MinHeap
    for i in range(n_items):
        for j in range(i + 1, n_items):
            theta = compute_ordering_exchange_2d(data, i, j)
            if theta is not None:
                exchanges.append((theta, i, j))

    # Add boundaries
    exchanges.append((0.0, -1, -1))
    exchanges.append((np.pi/2, -1, -1))
    exchanges.sort()

    # Sweep through angles and identify regions
    regions = [] # Should be MaxHeap
    for k in range(len(exchanges) - 1):
        theta_start = exchanges[k][0]
        theta_end = exchanges[k + 1][0]

        if theta_end - theta_start < 1e-10:
            continue

        # Sample a weight in this region
        theta_mid = (theta_start + theta_end) / 2
        weights = np.array([np.cos(theta_mid), np.sin(theta_mid)])
        ranking = get_ranking(data, weights)

        # Verify this ranking
        stability, (t_min, t_max) = verify_stability_2d(data, ranking)

        if stability > 0:
            regions.append((ranking, stability, (t_min, t_max)))

    # Sort by stability and return top n
    regions.sort(key=lambda x: x[1], reverse=True)
    return regions[:n_top]