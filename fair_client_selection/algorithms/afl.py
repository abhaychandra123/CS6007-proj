"""
AFL: Agnostic Federated Learning
Based on: Mohri et al. "Agnostic Federated Learning" (ICML 2019)

Key idea: Optimize for worst-case client performance using adversarial reweighting.
Selection prioritizes clients with highest adversarial weights.
"""
import numpy as np
from typing import List, Dict
from .base import FairClientSelector


class AFLSelector(FairClientSelector):
    """
    Agnostic Federated Learning (AFL) client selection.
    
    Focuses on improving worst-case client performance by maintaining
    adversarial weights and prioritizing poorly-performing clients.
    """
    
    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        learning_rate: float = 0.01,  # For adversarial weight updates
        seed: int = 42
    ):
        """
        Args:
            num_clients: Total number of clients
            clients_per_round: Number of clients to select per round
            learning_rate: Learning rate for adversarial weight updates
            seed: Random seed
        """
        super().__init__(num_clients, clients_per_round, seed)
        self.learning_rate = learning_rate
        
        # Initialize uniform adversarial weights
        self.adversarial_weights = {i: 1.0 / num_clients for i in range(num_clients)}
        self.client_losses = {i: 1.0 for i in range(num_clients)}
        
    def select_clients(
        self,
        client_info: Dict[int, Dict],
        round_num: int
    ) -> List[int]:
        """
        Select clients based on adversarial weights.
        
        Clients with higher adversarial weights (worse performance) are more likely to be selected.
        """
        available_clients = list(client_info.keys())
        
        # Update client losses and adversarial weights
        for client_id, info in client_info.items():
            if 'loss' in info:
                old_loss = self.client_losses.get(client_id, 1.0)
                new_loss = max(info['loss'], 1e-6)
                self.client_losses[client_id] = new_loss
                
                # Update adversarial weight using multiplicative weights update
                # w_i = w_i * exp(lr * loss_i)
                old_weight = self.adversarial_weights.get(client_id, 1.0 / self.num_clients)
                new_weight = old_weight * np.exp(self.learning_rate * new_loss)
                self.adversarial_weights[client_id] = new_weight
        
        # Normalize adversarial weights
        total_weight = sum(self.adversarial_weights.values())
        if total_weight > 0:
            for client_id in self.adversarial_weights.keys():
                self.adversarial_weights[client_id] /= total_weight
        
        # Get weights for available clients
        weights = []
        for client_id in available_clients:
            weight = self.adversarial_weights.get(client_id, 1.0 / self.num_clients)
            weights.append(weight)
        
        weights = np.array(weights)
        
        # Normalize to probabilities
        if weights.sum() > 0:
            probs = weights / weights.sum()
        else:
            probs = np.ones(len(available_clients)) / len(available_clients)
        
        # Sample clients based on adversarial weights
        selected = self.rng.choice(
            available_clients,
            size=min(self.clients_per_round, len(available_clients)),
            replace=False,
            p=probs
        ).tolist()
        
        self.update_selection_history(selected)
        return selected
    
    def get_algorithm_name(self) -> str:
        return "AFL"
