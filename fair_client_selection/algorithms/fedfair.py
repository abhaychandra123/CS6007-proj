"""
FedFair: Fairness-aware Federated Learning
Based on: Ezzeldin et al. "FairFed: Enabling Group Fairness in Federated Learning" (2021)

Key idea: Reweight client contributions during aggregation to reduce bias,
and prioritize selection of underrepresented groups.
"""
import numpy as np
from typing import List, Dict
from .base import FairClientSelector


class FedFairSelector(FairClientSelector):
    """
    FedFair client selection for group fairness.
    
    Ensures fair representation of different demographic groups by tracking
    group participation and prioritizing underrepresented groups.
    """
    
    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        fairness_threshold: float = 0.1,  # Target max difference in group selection rates
        seed: int = 42
    ):
        """
        Args:
            num_clients: Total number of clients
            clients_per_round: Number of clients to select per round
            fairness_threshold: Maximum allowed difference in group selection rates
            seed: Random seed
        """
        super().__init__(num_clients, clients_per_round, seed)
        self.fairness_threshold = fairness_threshold
        self.group_selection_counts = {}  # Track selections per group
        self.group_total_counts = {}  # Track total clients per group
        
    def select_clients(
        self,
        client_info: Dict[int, Dict],
        round_num: int
    ) -> List[int]:
        """
        Select clients to balance group representation.
        
        Prioritizes groups with lower selection rates to achieve fairness.
        """
        available_clients = list(client_info.keys())
        
        # Update group counts
        for client_id, info in client_info.items():
            group = info.get('group', 0)  # Default group 0 if not specified
            if group not in self.group_total_counts:
                self.group_total_counts[group] = 0
                self.group_selection_counts[group] = 0
            self.group_total_counts[group] = self.group_total_counts.get(group, 0) + 1
        
        # Calculate current selection rates per group
        group_selection_rates = {}
        for group in self.group_selection_counts.keys():
            total = self.group_total_counts.get(group, 1)
            selections = self.group_selection_counts.get(group, 0)
            group_selection_rates[group] = selections / (round_num + 1) if round_num > 0 else 0
        
        # Assign weights inversely proportional to selection rate
        # (underrepresented groups get higher weight)
        weights = []
        client_groups = []
        
        for client_id in available_clients:
            info = client_info[client_id]
            group = info.get('group', 0)
            client_groups.append(group)
            
            # Weight inversely proportional to group selection rate
            rate = group_selection_rates.get(group, 0)
            # Add small constant to avoid division by zero
            weight = 1.0 / (rate + 0.1)
            weights.append(weight)
        
        weights = np.array(weights)
        
        # Normalize to probabilities
        if weights.sum() > 0:
            probs = weights / weights.sum()
        else:
            probs = np.ones(len(available_clients)) / len(available_clients)
        
        # Sample clients
        selected = self.rng.choice(
            available_clients,
            size=min(self.clients_per_round, len(available_clients)),
            replace=False,
            p=probs
        ).tolist()
        
        # Update group selection counts
        for idx, client_id in enumerate(selected):
            group = client_groups[available_clients.index(client_id)]
            self.group_selection_counts[group] = self.group_selection_counts.get(group, 0) + 1
        
        self.update_selection_history(selected)
        return selected
    
    def get_algorithm_name(self) -> str:
        return "FedFair"
