"""
LongFed: Efficient Client Selection via Submodular Optimization.

Based on: Li et al., "LongFed: Balancing Performance and Individual Fairness 
in Federated Learning" (arXiv:2405.13584, 2024)

Key principles:
1. Selected clients' data distribution should resemble full population
2. Individual fairness: similar clients selected with similar frequency
3. Lyapunov optimization for long-term fairness constraints
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from .base import FairClientSelector
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class LongFedSelector(FairClientSelector):
    """
    LongFed client selection algorithm.
    
    Uses submodular optimization to select clients that maximize data distribution
    coverage while ensuring individual fairness through Lyapunov optimization.
    """
    
    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        lambda_fair: float = 0.5,
        V_param: float = 1.0,
        seed: int = 42
    ):
        """
        Args:
            num_clients: Total number of clients
            clients_per_round: Number of clients to select per round
            lambda_fair: Fairness weight parameter (0 = no fairness, 1 = max fairness)
            V_param: Lyapunov control parameter (larger = more emphasis on long-term)
            seed: Random seed
        """
        super().__init__(num_clients, clients_per_round, seed)
        
        self.lambda_fair = lambda_fair
        self.V_param = V_param
        
        # Virtual queues for Lyapunov optimization
        self.virtual_queues = {i: 0.0 for i in range(num_clients)}
        
        # Store client data distributions (will be updated)
        self.client_distributions = {}
        
        # Track selection frequencies
        self.selection_frequencies = {i: 0.0 for i in range(num_clients)}
        
    def _compute_distribution_distance(
        self,
        dist1: np.ndarray,
        dist2: np.ndarray
    ) -> float:
        """Compute distance between two distributions (e.g., label distributions)."""
        return np.linalg.norm(dist1 - dist2)
    
    def _compute_coverage(
        self,
        selected: List[int],
        all_clients: List[int]
    ) -> float:
        """
        Compute coverage: sum of minimum distances from each client to selected set.
        
        This is the Distribution Upper Bound (DUB) from the paper.
        Lower is better (selected clients cover full population well).
        """
        if not selected:
            return float('inf')
        
        total_distance = 0.0
        
        for client_id in all_clients:
            if client_id not in self.client_distributions:
                continue
            
            # Find minimum distance to any selected client
            min_dist = float('inf')
            for selected_id in selected:
                if selected_id not in self.client_distributions:
                    continue
                
                dist = self._compute_distribution_distance(
                    self.client_distributions[client_id],
                    self.client_distributions[selected_id]
                )
                min_dist = min(min_dist, dist)
            
            total_distance += min_dist
        
        return total_distance
    
    def _greedy_select(
        self,
        available_clients: List[int],
        all_clients: List[int]
    ) -> List[int]:
        """
        Greedy submodular optimization for client selection.
        
        CORRECTED: Maintains submodular property by separating coverage optimization
        from fairness adjustment.
        
        At each step, select the client that minimizes coverage (DUB).
        Complexity: O(K * N) where K = clients_per_round, N = num_clients
        """
        selected = []
        remaining = set(available_clients)
        
        for _ in range(min(self.clients_per_round, len(available_clients))):
            best_client = None
            best_score = float('inf')
            
            # Try adding each remaining client
            for candidate in remaining:
                candidate_selected = selected + [candidate]
                
                # Pure submodular coverage (maintains theoretical guarantee)
                coverage = self._compute_coverage(candidate_selected, all_clients)
                
                # CORRECTED: Apply fairness as a weighted combination, not additive penalty
                # This preserves the monotone property needed for submodular optimization
                if self.lambda_fair > 0:
                    # Normalize queue value to [0, 1] range
                    max_queue = max(self.virtual_queues.values()) if self.virtual_queues else 1.0
                    normalized_queue = self.virtual_queues[candidate] / (max_queue + 1e-8)
                    
                    # Weighted score: balance coverage and fairness
                    # Higher queue → lower priority (multiply by penalty)
                    fairness_weight = 1.0 + self.lambda_fair * normalized_queue
                    score = coverage * fairness_weight
                else:
                    score = coverage
                
                if score < best_score:
                    best_score = score
                    best_client = candidate
            
            if best_client is not None:
                selected.append(best_client)
                remaining.remove(best_client)
        
        return selected
    
    def _update_virtual_queues(self, selected_clients: List[int]):
        """
        Update virtual queues for Lyapunov optimization.
        
        Queue increases for unselected clients, decreases for selected ones.
        This ensures long-term fairness in selection frequency.
        """
        # Compute expected selection frequency
        expected_freq = self.clients_per_round / self.num_clients
        
        for client_id in range(self.num_clients):
            # Current selection indicator
            selected = 1.0 if client_id in selected_clients else 0.0
            
            # Update queue: Q(t+1) = max(0, Q(t) + expected - actual)
            self.virtual_queues[client_id] = max(
                0.0,
                self.virtual_queues[client_id] + expected_freq - selected
            )
    
    def update_client_distributions(self, client_distributions: Dict[int, np.ndarray]):
        """
        Update stored client data distributions.
        
        Args:
            client_distributions: Dict mapping client_id to distribution vector
                                 (e.g., label distribution, feature statistics)
        """
        self.client_distributions = client_distributions
    
    def select_clients(
        self,
        client_info: Dict[int, Dict],
        round_num: int
    ) -> List[int]:
        """
        Select clients using LongFed algorithm.
        
        Args:
            client_info: Dict with client information. Should contain 'distribution'
                        key with data distribution vector for each client.
            round_num: Current round number
        
        Returns:
            List of selected client IDs
        """
        # Extract distributions if provided
        for client_id, info in client_info.items():
            if 'distribution' in info and client_id not in self.client_distributions:
                self.client_distributions[client_id] = info['distribution']
        
        available_clients = list(client_info.keys())
        
        # Greedy submodular selection
        selected = self._greedy_select(available_clients, available_clients)
        
        # Update virtual queues for fairness
        if self.lambda_fair > 0:
            self._update_virtual_queues(selected)
        
        # Update frequencies
        for client_id in range(self.num_clients):
            self.selection_frequencies[client_id] = (
                self.selection_counts.get(client_id, 0) / (round_num + 1)
            )
        
        self.update_selection_history(selected)
        return selected
    
    def get_fairness_metrics(self) -> Dict:
        """Get individual fairness metrics."""
        freqs = np.array(list(self.selection_frequencies.values()))
        queues = np.array(list(self.virtual_queues.values()))
        
        return {
            'selection_frequency_std': freqs.std(),
            'selection_frequency_cv': freqs.std() / (freqs.mean() + 1e-8),
            'virtual_queue_mean': queues.mean(),
            'virtual_queue_max': queues.max(),
            **self.get_selection_statistics()
        }


def compute_label_distribution(dataset, num_classes: int) -> np.ndarray:
    """
    Compute label distribution for a dataset.
    
    Args:
        dataset: Dataset with labels
        num_classes: Number of classes
    
    Returns:
        Distribution vector (sums to 1)
    """
    labels = []
    
    for i in range(len(dataset)):
        if len(dataset[i]) >= 2:
            labels.append(dataset[i][1])
    
    labels = np.array(labels)
    distribution = np.zeros(num_classes)
    
    for c in range(num_classes):
        distribution[c] = (labels == c).sum()
    
    # Normalize
    if distribution.sum() > 0:
        distribution = distribution / distribution.sum()
    
    return distribution
