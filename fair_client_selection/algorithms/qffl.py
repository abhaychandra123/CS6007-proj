"""
q-FFL: Fair Resource Allocation in Federated Learning
Based on: Li et al. "Fair Resource Allocation in Federated Learning" (ICLR 2020)

Key idea: Use q-FFL loss that penalizes high-performing clients to improve tail performance.
Selection prioritizes clients with higher loss (worse performance).
"""
import numpy as np
from typing import List, Dict
from .base import FairClientSelector


class QFFLSelector(FairClientSelector):
    """
    q-FFL (q-Fair Federated Learning) client selection.
    
    Prioritizes clients with higher losses to improve fairness in model performance
    across clients. The parameter q controls the fairness-accuracy trade-off.
    """
    
    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        q: float = 5.0,  # Fairness parameter (higher = more fair, lower = more accurate)
        seed: int = 42
    ):
        """
        Args:
            num_clients: Total number of clients
            clients_per_round: Number of clients to select per round
            q: Fairness parameter. q=0 is standard FL, q→∞ prioritizes worst performers
            seed: Random seed
        """
        super().__init__(num_clients, clients_per_round, seed)
        self.q = q
        self.client_losses = {i: 1.0 for i in range(num_clients)}  # Initialize to 1.0
        
    def select_clients(
        self,
        client_info: Dict[int, Dict],
        round_num: int
    ) -> List[int]:
        """
        Select clients based on weighted sampling proportional to loss^q.
        
        Higher loss clients get higher selection probability.
        """
        available_clients = list(client_info.keys())
        
        # Update client losses if available
        for client_id, info in client_info.items():
            if 'loss' in info:
                self.client_losses[client_id] = max(info['loss'], 1e-6)
        
        # Compute q-FFL weights: loss^q
        weights = []
        for client_id in available_clients:
            loss = self.client_losses.get(client_id, 1.0)
            weight = loss ** self.q
            weights.append(weight)
        
        weights = np.array(weights)
        
        # Normalize to probabilities
        if weights.sum() > 0:
            probs = weights / weights.sum()
        else:
            probs = np.ones(len(available_clients)) / len(available_clients)
        
        # Sample clients based on probabilities
        selected = self.rng.choice(
            available_clients,
            size=min(self.clients_per_round, len(available_clients)),
            replace=False,
            p=probs
        ).tolist()
        
        self.update_selection_history(selected)
        return selected
    
    def get_algorithm_name(self) -> str:
        return f"q-FFL(q={self.q})"
