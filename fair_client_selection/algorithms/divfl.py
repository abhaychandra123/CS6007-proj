"""
DivFL: Diversity-based Fair Federated Learning
Based on diversity maximization principles for fair client selection.

Key idea: Select diverse clients to maximize coverage of data distribution
while maintaining fairness through diversity metrics.
"""
import numpy as np
from typing import List, Dict
from .base import FairClientSelector


class DivFLSelector(FairClientSelector):
    """
    Diversity-based Fair FL client selection.
    
    Selects clients to maximize diversity in data distribution and group representation
    while ensuring fair participation.
    """
    
    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        diversity_weight: float = 0.5,  # Balance between diversity and fairness
        seed: int = 42
    ):
        """
        Args:
            num_clients: Total number of clients
            clients_per_round: Number of clients to select per round
            diversity_weight: Weight for diversity (vs participation fairness)
            seed: Random seed
        """
        super().__init__(num_clients, clients_per_round, seed)
        self.diversity_weight = diversity_weight
        self.client_embeddings = {}  # Store client data distribution embeddings
        
    def select_clients(
        self,
        client_info: Dict[int, Dict],
        round_num: int
    ) -> List[int]:
        """
        Select clients to maximize diversity + fairness.
        
        Uses greedy selection to maximize coverage while penalizing over-selected clients.
        """
        available_clients = list(client_info.keys())
        
        # Update client embeddings if available
        for client_id, info in client_info.items():
            if 'embedding' in info:
                self.client_embeddings[client_id] = np.array(info['embedding'])
            elif 'data_distribution' in info:
                # Use data distribution as embedding
                self.client_embeddings[client_id] = np.array(info['data_distribution'])
        
        # If no embeddings available, use random features
        if not self.client_embeddings:
            for client_id in available_clients:
                self.client_embeddings[client_id] = self.rng.randn(10)  # Random 10-d embedding
        
        # Greedy selection for diversity
        selected = []
        selected_embeddings = []
        
        for _ in range(min(self.clients_per_round, len(available_clients))):
            best_client = None
            best_score = -np.inf
            
            for client_id in available_clients:
                if client_id in selected:
                    continue
                
                # Diversity score: distance to selected clients
                embedding = self.client_embeddings.get(client_id, self.rng.randn(10))
                
                if len(selected_embeddings) == 0:
                    diversity_score = 1.0  # First client gets max score
                else:
                    # Average distance to selected clients
                    distances = [
                        np.linalg.norm(embedding - sel_emb)
                        for sel_emb in selected_embeddings
                    ]
                    diversity_score = np.mean(distances)
                
                # Fairness score: inverse of selection count
                selection_count = self.selection_counts.get(client_id, 0)
                fairness_score = 1.0 / (selection_count + 1)
                
                # Combined score
                score = (
                    self.diversity_weight * diversity_score +
                    (1 - self.diversity_weight) * fairness_score
                )
                
                if score > best_score:
                    best_score = score
                    best_client = client_id
            
            if best_client is not None:
                selected.append(best_client)
                selected_embeddings.append(
                    self.client_embeddings.get(best_client, self.rng.randn(10))
                )
        
        self.update_selection_history(selected)
        return selected
    
    def get_algorithm_name(self) -> str:
        return f"DivFL(α={self.diversity_weight})"
