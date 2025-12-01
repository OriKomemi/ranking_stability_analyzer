import numpy as np
from typing import List, Tuple, Optional


class RankingStabilityAnalyzer:
    """
    Core class implementing ranking stability algorithms from the paper.

    The analyzer operates in two modes:
    - 2D (exact): Uses geometric algorithms for precise stability computation
    """

    def __init__(self, data: np.ndarray, attribute_names: List[str] = None):
        """
        Initialize the analyzer with a dataset.

        Args:
            data: n x d numpy array where n is number of items, d is number of attributes
            attribute_names: List of attribute names
        """
        self.data = data
        self.n_items, self.n_attrs = data.shape
        self.attribute_names = attribute_names or [f"Attr_{i+1}" for i in range(self.n_attrs)]

    def compute_score(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute scores for all items given a weight vector.

        Implements the scoring function: score(t_i) = w · t_i (Equation 1)

        Args:
            weights: d-dimensional weight vector

        Returns:
            Array of scores for each item
        """
        return self.data @ weights

    def get_ranking(self, weights: np.ndarray) -> np.ndarray:
        """
        Get ranking of items based on weights (higher score = better rank).

        Args:
            weights: d-dimensional weight vector

        Returns:
            Array of item indices in ranked order (best to worst)
        """
        scores = self.compute_score(weights)
        return np.argsort(-scores)  # Descending order

    def compute_ordering_exchange_2d(self, item_i: int, item_j: int) -> Optional[float]:
        """
        Compute the ordering exchange angle between two items in 2D.
        Based on Equation 6 in the paper.

        The ordering exchange is the angle θ where two items have equal scores,
        representing the boundary where they swap positions in the ranking.

        Args:
            item_i, item_j: Indices of two items

        Returns:
            Angle theta where items exchange order (None if one dominates)
        """
        if self.n_attrs != 2:
            raise ValueError("Ordering exchange angle only defined for 2D")

        t_i = self.data[item_i]
        t_j = self.data[item_j]

        # Check for dominance
        if np.all(t_i >= t_j) and np.any(t_i > t_j):
            return None  # i dominates j
        if np.all(t_j >= t_i) and np.any(t_j > t_i):
            return None  # j dominates i

        # Compute ordering exchange angle (Equation 6)
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

    def verify_stability_2d(self, ranking: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """
        Verify the stability of a given ranking in 2D.
        Algorithm from Section 3.1 of the paper.

        Computes the exact stability S_D(r) = vol(R_D(r)) / vol(U) by finding
        the angular range [θ_min, θ_max] where the ranking is valid.

        Args:
            ranking: Array of item indices in ranked order

        Returns:
            Tuple of (stability, (theta_min, theta_max))
        """
        if self.n_attrs != 2:
            raise ValueError("2D stability verification only works for 2D data")

        theta_min = 0.0
        theta_max = np.pi / 2

        # Check each adjacent pair in the ranking
        for i in range(len(ranking) - 1):
            item_higher = ranking[i]
            item_lower = ranking[i + 1]

            theta_exchange = self.compute_ordering_exchange_2d(item_higher, item_lower)

            if theta_exchange is None:
                # Check if ranking is valid
                if not self.dominates(item_higher, item_lower):
                    return 0.0, (0.0, 0.0)  # Invalid ranking
                continue

            # Determine which side of the exchange the ranking is valid
            if self.data[item_higher, 0] < self.data[item_lower, 0]:
                # Higher ranked item has smaller x1, so valid for theta < theta_exchange
                theta_max = min(theta_max, theta_exchange)
            else:
                # Higher ranked item has larger x1, so valid for theta > theta_exchange
                theta_min = max(theta_min, theta_exchange)

        if theta_min >= theta_max:
            return 0.0, (0.0, 0.0)

        # Stability is the proportion of the angular range [0, π/2]
        stability = (theta_max - theta_min) / (np.pi / 2)
        return stability, (theta_min, theta_max)

    def dominates(self, item_i: int, item_j: int) -> bool:
        """
        Check if item_i dominates item_j.

        Item i dominates item j if i is better or equal on all attributes,
        and strictly better on at least one attribute.
        """
        return np.all(self.data[item_i] >= self.data[item_j]) and \
               np.any(self.data[item_i] > self.data[item_j])

    def sample_uniform_weights(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample weight vectors uniformly from the function space.
        Based on Section 5.1 of the paper.

        Samples from the positive orthant of the unit sphere to ensure
        uniform distribution over all valid scoring functions.

        Args:
            n_samples: Number of samples to generate

        Returns:
            n_samples x d array of weight vectors
        """
        # Sample from standard normal and take absolute value
        weights = np.abs(np.random.randn(n_samples, self.n_attrs))
        # Normalize to unit length
        weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
        return weights

    def enumerate_stable_rankings_2d(self, n_top: int = 10) -> List[Tuple[np.ndarray, float, Tuple[float, float]]]:
        """
        Enumerate the most stable rankings in 2D using ray sweeping.
        Based on Algorithm 1 (RAYSWEEPING) from the paper (§3.2).

        This algorithm sweeps through all ordering exchanges to identify
        distinct ranking regions and computes their exact stability.

        Time complexity: O(n²log n) where n is the number of items.

        Args:
            n_top: Number of top stable rankings to return

        Returns:
            List of tuples (ranking, stability, (theta_min, theta_max))
        """
        if self.n_attrs != 2:
            raise ValueError("2D enumeration only works for 2D data")

        # Compute all ordering exchanges
        exchanges = []
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                theta = self.compute_ordering_exchange_2d(i, j)
                if theta is not None:
                    exchanges.append((theta, i, j))

        # Add boundaries
        exchanges.append((0.0, -1, -1))
        exchanges.append((np.pi/2, -1, -1))
        exchanges.sort()

        # Sweep through angles and identify regions
        regions = []
        for k in range(len(exchanges) - 1):
            theta_start = exchanges[k][0]
            theta_end = exchanges[k + 1][0]

            if theta_end - theta_start < 1e-10:
                continue

            # Sample a weight in this region
            theta_mid = (theta_start + theta_end) / 2
            weights = np.array([np.cos(theta_mid), np.sin(theta_mid)])
            ranking = self.get_ranking(weights)

            # Verify this ranking
            stability, (t_min, t_max) = self.verify_stability_2d(ranking)

            if stability > 0:
                regions.append((ranking, stability, (t_min, t_max)))

        # Sort by stability and return top n
        regions.sort(key=lambda x: x[1], reverse=True)
        return regions[:n_top]

    def find_top_k_stable_rankings(self, k: int = 10, n_samples: int = 5000,
                                   n_rankings: int = 10) -> List[Tuple[np.ndarray, float, float]]:
        """
        Find the most stable top-k rankings using randomized sampling.

        This method is useful for multi-dimensional datasets where exact
        enumeration is computationally expensive. It uses Monte Carlo sampling
        to discover the most frequently occurring top-k rankings.

        Args:
            k: Length of top-k list
            n_samples: Number of Monte Carlo samples
            n_rankings: Number of top rankings to return

        Returns:
            List of tuples (ranking, estimated_stability, confidence_error)
        """
        ranking_counts = {}

        for _ in range(n_samples):
            weights = self.sample_uniform_weights(1)[0]
            full_ranking = self.get_ranking(weights)
            top_k_ranking = tuple(full_ranking[:k])

            ranking_counts[top_k_ranking] = ranking_counts.get(top_k_ranking, 0) + 1

        # Compute stability estimates
        results = []
        for ranking, count in ranking_counts.items():
            stability = count / n_samples
            z_score = 1.96
            if stability > 0 and stability < 1:
                error = z_score * np.sqrt(stability * (1 - stability) / n_samples)
            else:
                error = 0.0
            results.append((np.array(ranking), stability, error))

        # Sort by stability
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_rankings]
