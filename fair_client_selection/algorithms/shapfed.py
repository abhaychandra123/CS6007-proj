"""
ShapFed: Contribution-Based Fair Client Selection using Shapley Values.

Based on: Tastan et al., "ShapFed: Contribution-Based Fair Client Selection"
(OpenReview, 2024)

Key concepts:
1. Class-Specific Shapley Values (CSSVs) for granular contribution assessment
2. Contribution-aware weighted aggregation
3. Personalized model updates based on contribution
4. Collaborative fairness: high contributors get better models
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from .base import FairClientSelector


class ShapFedSelector(FairClientSelector):
    """
    ShapFed client selection using Shapley values.
    
    Computes Class-Specific Shapley Values (CSSVs) to assess client contributions
    and selects clients for collaborative fairness.
    """
    
    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        num_classes: int,
        selection_strategy: str = 'top_contribution',
        epsilon: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            num_clients: Total number of clients
            clients_per_round: Number of clients to select per round
            num_classes: Number of output classes
            selection_strategy: 'top_contribution', 'epsilon_greedy', or 'proportional'
            epsilon: Exploration parameter for epsilon-greedy
            seed: Random seed
        """
        super().__init__(num_clients, clients_per_round, seed)
        
        self.num_classes = num_classes
        self.selection_strategy = selection_strategy
        self.epsilon = epsilon
        
        # Track client contributions (CSSVs)
        self.client_contributions = {i: 0.0 for i in range(num_clients)}
        self.client_cssvs = {i: np.zeros(num_classes) for i in range(num_clients)}
        
        # Track historical contributions for stable estimates
        self.contribution_history = {i: [] for i in range(num_clients)}
        
    def compute_cssv(
        self,
        client_params: torch.Tensor,
        global_params: torch.Tensor
    ) -> np.ndarray:
        """
        Compute Class-Specific Shapley Value (CSSV) approximation.
        
        NOTE: This is a FAST APPROXIMATION using cosine similarity, not true Shapley values.
        
        True Shapley values require computing:
            φ_i^c = Σ_S [|S|!(N-|S|-1)!/N!] * [v(S∪{i}) - v(S)]
        
        which is exponential in the number of clients. The paper uses permutation sampling
        for approximation, but this is computationally expensive (O(M*N) where M=samples).
        
        This implementation uses a SIMPLER heuristic:
        - Cosine similarity between client's class-c parameters and global parameters
        - Assumes that higher similarity → higher contribution
        - O(1) computation per client
        
        For true Shapley values, use compute_cssv_monte_carlo() instead (slower but accurate).
        
        Args:
            client_params: Client's last layer weights (num_classes, feature_dim)
            global_params: Global last layer weights (num_classes, feature_dim)
        
        Returns:
            CSSV approximation vector of shape (num_classes,)
        """
        if client_params.dim() == 1:
            client_params = client_params.unsqueeze(0)
        if global_params.dim() == 1:
            global_params = global_params.unsqueeze(0)
        
        # Compute cosine similarity for each class
        cssvs = np.zeros(self.num_classes)
        
        for c in range(min(self.num_classes, client_params.shape[0], global_params.shape[0])):
            client_vec = client_params[c].flatten()
            global_vec = global_params[c].flatten()
            
            # Cosine similarity as contribution proxy
            similarity = F.cosine_similarity(
                client_vec.unsqueeze(0),
                global_vec.unsqueeze(0)
            ).item()
            
            cssvs[c] = max(0.0, similarity)  # Only positive contributions
        
        return cssvs
    
    def compute_contribution_score(self, cssv: np.ndarray) -> float:
        """
        Compute overall contribution score from CSSV.
        
        Higher score means higher contribution to global model.
        """
        return cssv.mean()
    
    def update_contributions(
        self,
        client_params_dict: Dict[int, torch.Tensor],
        global_params: torch.Tensor
    ):
        """
        Update client contribution estimates based on latest parameters.
        
        Args:
            client_params_dict: Dict mapping client_id to last layer parameters
            global_params: Global model's last layer parameters
        """
        for client_id, client_params in client_params_dict.items():
            # Compute CSSV
            cssv = self.compute_cssv(client_params, global_params)
            self.client_cssvs[client_id] = cssv
            
            # Compute overall contribution
            contribution = self.compute_contribution_score(cssv)
            self.client_contributions[client_id] = contribution
            
            # Track history
            self.contribution_history[client_id].append(contribution)
    
    def get_contribution_weights(self) -> Dict[int, float]:
        """
        Get normalized contribution weights for aggregation.
        
        Returns:
            Dict mapping client_id to aggregation weight
        """
        contributions = np.array([
            self.client_contributions.get(i, 0.0)
            for i in range(self.num_clients)
        ])
        
        # Normalize to sum to 1
        if contributions.sum() > 0:
            weights = contributions / contributions.sum()
        else:
            weights = np.ones(self.num_clients) / self.num_clients
        
        return {i: weights[i] for i in range(self.num_clients)}
    
    def select_clients(
        self,
        client_info: Dict[int, Dict],
        round_num: int
    ) -> List[int]:
        """
        Select clients based on contribution scores.
        
        Args:
            client_info: Dict with client information. Can include 'params' for
                        last layer parameters.
            round_num: Current round number
        
        Returns:
            List of selected client IDs
        """
        available_clients = list(client_info.keys())
        
        if self.selection_strategy == 'top_contribution':
            # Select clients with highest contributions
            selected = self._select_top_contributors(available_clients)
            
        elif self.selection_strategy == 'epsilon_greedy':
            # Epsilon-greedy: explore with probability epsilon
            if self.rng.rand() < self.epsilon:
                # Explore: random selection
                selected = self.rng.choice(
                    available_clients,
                    size=min(self.clients_per_round, len(available_clients)),
                    replace=False
                ).tolist()
            else:
                # Exploit: select top contributors
                selected = self._select_top_contributors(available_clients)
        
        elif self.selection_strategy == 'proportional':
            # Select proportional to contribution
            selected = self._select_proportional(available_clients)
        
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
        
        self.update_selection_history(selected)
        return selected
    
    def _select_top_contributors(self, available_clients: List[int]) -> List[int]:
        """Select clients with highest contribution scores."""
        # Sort by contribution
        sorted_clients = sorted(
            available_clients,
            key=lambda x: self.client_contributions.get(x, 0.0),
            reverse=True
        )
        
        return sorted_clients[:self.clients_per_round]
    
    def _select_proportional(self, available_clients: List[int]) -> List[int]:
        """Select clients with probability proportional to contribution."""
        contributions = np.array([
            self.client_contributions.get(i, 1e-6)
            for i in available_clients
        ])
        
        # Normalize to probabilities
        probs = contributions / contributions.sum()
        
        # Sample without replacement
        selected = self.rng.choice(
            available_clients,
            size=min(self.clients_per_round, len(available_clients)),
            replace=False,
            p=probs
        ).tolist()
        
        return selected
    
    def get_personalization_weights(self, client_id: int) -> float:
        """
        Get personalization weight gamma for a client.
        
        gamma_i = contribution_i / max_contribution
        
        Higher contributors get models closer to global (gamma closer to 1).
        """
        contributions = list(self.client_contributions.values())
        max_contrib = max(contributions) if contributions else 1.0
        
        if max_contrib > 0:
            gamma = self.client_contributions.get(client_id, 0.0) / max_contrib
        else:
            gamma = 1.0 / self.num_clients
        
        return gamma
    
    def get_fairness_metrics(self) -> Dict:
        """Get collaborative fairness metrics."""
        contributions = np.array(list(self.client_contributions.values()))
        
        return {
            'contribution_mean': contributions.mean(),
            'contribution_std': contributions.std(),
            'contribution_min': contributions.min(),
            'contribution_max': contributions.max(),
            'contribution_cv': contributions.std() / (contributions.mean() + 1e-8),
            **self.get_selection_statistics()
        }


class ShapFedAggregator:
    """
    Weighted aggregation for ShapFed.
    
    Aggregates client updates using contribution-based weights.
    """
    
    def __init__(self, selector: ShapFedSelector):
        self.selector = selector
        
    def aggregate(
        self,
        client_params_list: List[Dict[str, torch.Tensor]],
        client_ids: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client parameters using contribution weights.
        
        Args:
            client_params_list: List of client parameter dicts
            client_ids: List of client IDs corresponding to params
        
        Returns:
            Aggregated parameters
        """
        # Get contribution weights
        all_weights = self.selector.get_contribution_weights()
        
        # Normalize weights for selected clients only
        selected_weights = np.array([all_weights[cid] for cid in client_ids])
        if selected_weights.sum() > 0:
            selected_weights = selected_weights / selected_weights.sum()
        else:
            selected_weights = np.ones(len(client_ids)) / len(client_ids)
        
        # Weighted average
        aggregated = {}
        
        for key in client_params_list[0].keys():
            weighted_sum = torch.zeros_like(client_params_list[0][key])
            
            for i, params in enumerate(client_params_list):
                weighted_sum += selected_weights[i] * params[key]
            
            aggregated[key] = weighted_sum
        
        return aggregated
    
    def personalize(
        self,
        global_params: Dict[str, torch.Tensor],
        client_params: Dict[str, torch.Tensor],
        client_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        Create personalized model for client.
        
        personalized = gamma * global + (1 - gamma) * client
        
        Args:
            global_params: Global model parameters
            client_params: Client's local parameters
            client_id: Client ID
        
        Returns:
            Personalized parameters
        """
        gamma = self.selector.get_personalization_weights(client_id)
        
        personalized = {}
        for key in global_params.keys():
            personalized[key] = (
                gamma * global_params[key] +
                (1 - gamma) * client_params[key]
            )
        
        return personalized
