"""
MAB Fair Selection: Multi-Armed Bandit Framework for Demographic Fairness.

Based on: Bouzamoucha et al., "Multi-Armed Bandit Approach to Fair Client 
Selection" (ICAART 2025)

Key concepts:
1. Each client is an "arm" in multi-armed bandit
2. Reward based on improvement in global fairness metrics (SPD, EOD)
3. Epsilon-greedy exploration-exploitation balance
4. Privacy-preserving: uses local fairness metrics
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from .base import FairClientSelector


class MABFairSelector(FairClientSelector):
    """
    Multi-Armed Bandit based fair client selection.
    
    Frames client selection as MAB problem where rewards are based on
    fairness improvements.
    """
    
    def __init__(
        self,
        num_clients: int,
        clients_per_round: int,
        epsilon: float = 0.2,
        alpha_fair: float = 1.0,
        beta_fair: float = 1.0,
        gamma_acc: float = 0.5,
        fairness_metric: str = 'spd',
        seed: int = 42
    ):
        """
        Args:
            num_clients: Total number of clients
            clients_per_round: Number of clients to select per round
            epsilon: Exploration parameter (0 = exploit only, 1 = explore only)
            alpha_fair: Weight for global fairness in reward
            beta_fair: Weight for local fairness in reward
            gamma_acc: Weight for accuracy in reward
            fairness_metric: 'spd' or 'eod'
            seed: Random seed
        """
        super().__init__(num_clients, clients_per_round, seed)
        
        self.epsilon = epsilon
        self.alpha_fair = alpha_fair
        self.beta_fair = beta_fair
        self.gamma_acc = gamma_acc
        self.fairness_metric = fairness_metric
        
        # Track rewards for each arm (client)
        self.mean_rewards = {i: 0.0 for i in range(num_clients)}
        self.reward_counts = {i: 0 for i in range(num_clients)}
        self.reward_history = {i: [] for i in range(num_clients)}
        
        # Track global metrics
        self.global_fairness_history = []
        self.global_accuracy_history = []
        
        # Store client fairness metrics
        self.client_fairness = {i: 1.0 for i in range(num_clients)}  # Initialize to 1 (worst)
        
    def compute_reward(
        self,
        client_id: int,
        global_fairness_new: float,
        global_fairness_old: float,
        global_acc_new: float,
        global_acc_old: float,
        local_fairness: float
    ) -> float:
        """
        Compute reward for selecting a client.
        
        CORRECTED FORMULA (from paper):
        R_i = α * (ΔF_global / (β * F_local,i + ε)) + γ * ΔAcc_global
        
        Where:
        - ΔF_global = improvement in global fairness (old - new, since lower is better)
        - F_local,i = local fairness metric of client i (SPD or EOD)
        - ΔAcc_global = improvement in global accuracy (new - old)
        - ε = small constant to prevent division by zero
        
        Args:
            client_id: Client ID
            global_fairness_new: Global fairness after aggregation (SPD/EOD)
            global_fairness_old: Global fairness before aggregation (SPD/EOD)
            global_acc_new: Global accuracy after aggregation
            global_acc_old: Global accuracy before aggregation
            local_fairness: Client's local fairness metric (SPD/EOD)
        
        Returns:
            Reward value (higher is better)
        """
        # Global fairness improvement
        # Since SPD/EOD are in [0,1] where 0 is perfectly fair:
        # Improvement = old_fairness - new_fairness (positive means improvement)
        delta_fairness_global = global_fairness_old - global_fairness_new
        
        # Accuracy improvement
        delta_acc_global = global_acc_new - global_acc_old
        
        # Paper formula: divide fairness improvement by local fairness
        # Intuition: Clients with lower local fairness (more fair) contribute more to global fairness
        fairness_term = self.alpha_fair * (
            delta_fairness_global / (self.beta_fair * local_fairness + 1e-6)
        )
        
        # Accuracy term
        acc_term = self.gamma_acc * delta_acc_global
        
        # Combined reward
        reward = fairness_term + acc_term
        
        return reward
    
    def update_rewards(
        self,
        selected_clients: List[int],
        global_fairness_new: float,
        global_fairness_old: float,
        global_acc_new: float,
        global_acc_old: float,
        client_fairness_dict: Dict[int, float]
    ):
        """
        Update reward estimates for selected clients.
        
        Args:
            selected_clients: List of client IDs that were selected
            global_fairness_new: New global fairness metric
            global_fairness_old: Old global fairness metric
            global_acc_new: New global accuracy
            global_acc_old: Old global accuracy
            client_fairness_dict: Dict mapping client_id to local fairness
        """
        # Update client fairness metrics
        self.client_fairness.update(client_fairness_dict)
        
        # Compute reward for each selected client
        for client_id in selected_clients:
            local_fairness = client_fairness_dict.get(client_id, 1.0)
            
            reward = self.compute_reward(
                client_id,
                global_fairness_new,
                global_fairness_old,
                global_acc_new,
                global_acc_old,
                local_fairness
            )
            
            # Update running average of rewards (incremental mean)
            self.reward_counts[client_id] += 1
            n = self.reward_counts[client_id]
            old_mean = self.mean_rewards[client_id]
            self.mean_rewards[client_id] = old_mean + (reward - old_mean) / n
            
            # Store in history
            self.reward_history[client_id].append(reward)
        
        # Track global metrics
        self.global_fairness_history.append(global_fairness_new)
        self.global_accuracy_history.append(global_acc_new)
    
    def select_clients(
        self,
        client_info: Dict[int, Dict],
        round_num: int
    ) -> List[int]:
        """
        Select clients using epsilon-greedy strategy.
        
        Args:
            client_info: Dict with client information. Should contain 'fairness'
                        for local fairness metrics.
            round_num: Current round number
        
        Returns:
            List of selected client IDs
        """
        available_clients = list(client_info.keys())
        
        # Epsilon-greedy selection
        if self.rng.rand() < self.epsilon:
            # Explore: random selection
            selected = self.rng.choice(
                available_clients,
                size=min(self.clients_per_round, len(available_clients)),
                replace=False
            ).tolist()
        else:
            # Exploit: select top-K by mean reward
            selected = self._select_top_reward(available_clients)
        
        self.update_selection_history(selected)
        return selected
    
    def _select_top_reward(self, available_clients: List[int]) -> List[int]:
        """Select clients with highest mean rewards."""
        # Sort clients by mean reward
        sorted_clients = sorted(
            available_clients,
            key=lambda x: self.mean_rewards.get(x, 0.0),
            reverse=True
        )
        
        return sorted_clients[:self.clients_per_round]
    
    def select_by_fairness(
        self,
        available_clients: List[int],
        strategy: str = 'lowest'
    ) -> List[int]:
        """
        Select clients based on local fairness metrics.
        
        Args:
            available_clients: List of available client IDs
            strategy: 'lowest' (most fair), 'highest' (least fair), or 'optimal' (closest to 0)
        
        Returns:
            List of selected client IDs
        """
        if strategy == 'lowest':
            # Select clients with lowest (best) fairness metrics
            sorted_clients = sorted(
                available_clients,
                key=lambda x: self.client_fairness.get(x, 1.0)
            )
        elif strategy == 'highest':
            # Select clients with highest (worst) fairness metrics
            sorted_clients = sorted(
                available_clients,
                key=lambda x: self.client_fairness.get(x, 0.0),
                reverse=True
            )
        elif strategy == 'optimal':
            # Select clients closest to optimal fairness (0)
            sorted_clients = sorted(
                available_clients,
                key=lambda x: abs(self.client_fairness.get(x, 1.0))
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return sorted_clients[:self.clients_per_round]
    
    def get_fairness_metrics(self) -> Dict:
        """Get MAB-specific fairness metrics."""
        rewards = np.array(list(self.mean_rewards.values()))
        counts = np.array(list(self.reward_counts.values()))
        
        metrics = {
            'mean_reward': rewards.mean(),
            'reward_std': rewards.std(),
            'reward_min': rewards.min(),
            'reward_max': rewards.max(),
            'exploration_rate': self.epsilon,
            **self.get_selection_statistics()
        }
        
        # Add global metrics if available
        if self.global_fairness_history:
            metrics['global_fairness_final'] = self.global_fairness_history[-1]
            metrics['global_fairness_improvement'] = (
                self.global_fairness_history[0] - self.global_fairness_history[-1]
                if len(self.global_fairness_history) > 1 else 0.0
            )
        
        if self.global_accuracy_history:
            metrics['global_accuracy_final'] = self.global_accuracy_history[-1]
            metrics['global_accuracy_improvement'] = (
                self.global_accuracy_history[-1] - self.global_accuracy_history[0]
                if len(self.global_accuracy_history) > 1 else 0.0
            )
        
        return metrics
    
    def update_epsilon(self, new_epsilon: float):
        """Update exploration parameter (can decay over time)."""
        self.epsilon = max(0.0, min(1.0, new_epsilon))
    
    def get_client_ranking(self) -> List[Tuple[int, float]]:
        """
        Get clients ranked by mean reward.
        
        Returns:
            List of (client_id, mean_reward) tuples, sorted descending
        """
        ranking = [(cid, self.mean_rewards[cid]) for cid in range(self.num_clients)]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
