"""Base classes for fair client selection algorithms."""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FairClientSelector(ABC):
    """Abstract base class for fair client selection algorithms."""
    
    def __init__(self, num_clients: int, clients_per_round: int, seed: int = 42):
        """
        Args:
            num_clients: Total number of clients
            clients_per_round: Number of clients to select per round
            seed: Random seed
        
        Raises:
            ValueError: If num_clients <= 0 or clients_per_round <= 0 or clients_per_round > num_clients
        """
        # Input validation
        if not isinstance(num_clients, int) or num_clients <= 0:
            raise ValueError(f"num_clients must be a positive integer, got {num_clients}")
        if not isinstance(clients_per_round, int) or clients_per_round <= 0:
            raise ValueError(f"clients_per_round must be a positive integer, got {clients_per_round}")
        if clients_per_round > num_clients:
            raise ValueError(f"clients_per_round ({clients_per_round}) cannot exceed num_clients ({num_clients})")
        
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        logger.debug(f"Initialized {self.__class__.__name__} with num_clients={num_clients}, clients_per_round={clients_per_round}")
        
        # Track selection history
        self.selection_history = []
        self.selection_counts = {i: 0 for i in range(num_clients)}
        self.current_round = 0
        
    @abstractmethod
    def select_clients(
        self,
        client_info: Dict[int, Dict],
        round_num: int
    ) -> List[int]:
        """
        Select clients for the current round.
        
        Args:
            client_info: Dictionary mapping client_id to client information
                         (e.g., fairness metrics, data distribution, etc.)
            round_num: Current round number
        
        Returns:
            List of selected client IDs
        """
        pass
    
    def update_selection_history(self, selected_clients: List[int]):
        """Update selection history and counts.
        
        Args:
            selected_clients: List of selected client IDs
        
        Raises:
            ValueError: If selected_clients contains invalid client IDs
        """
        if not isinstance(selected_clients, list):
            raise ValueError(f"selected_clients must be a list, got {type(selected_clients)}")
        
        # Validate all client IDs are within range
        for client_id in selected_clients:
            if not isinstance(client_id, (int, np.integer)):
                raise ValueError(f"Client ID must be an integer, got {type(client_id)}")
            if client_id < 0 or client_id >= self.num_clients:
                raise ValueError(f"Invalid client_id {client_id}, must be in range [0, {self.num_clients})")
        
        self.selection_history.append(selected_clients)
        for client_id in selected_clients:
            self.selection_counts[client_id] += 1
        self.current_round += 1
    
    def get_selection_statistics(self) -> Dict:
        """Get statistics about client selection.
        
        Returns:
            Dictionary with selection statistics
        
        Raises:
            ValueError: If no selections have been made yet
        """
        if self.current_round == 0:
            logger.warning("No selections made yet, returning default statistics")
            return {
                'mean_selections': 0.0,
                'std_selections': 0.0,
                'min_selections': 0,
                'max_selections': 0,
                'cv_selections': 0.0,
            }
        
        counts = np.array(list(self.selection_counts.values()))
        mean_val = counts.mean()
        std_val = counts.std()
        
        # Avoid division by zero for CV calculation
        cv_val = std_val / mean_val if mean_val > 1e-10 else 0.0
        
        return {
            'mean_selections': float(mean_val),
            'std_selections': float(std_val),
            'min_selections': int(counts.min()),
            'max_selections': int(counts.max()),
            'cv_selections': float(cv_val),
        }


class RandomSelector(FairClientSelector):
    """Baseline: Random client selection."""
    
    def select_clients(
        self,
        client_info: Dict[int, Dict],
        round_num: int
    ) -> List[int]:
        """Randomly select clients.
        
        Args:
            client_info: Dictionary mapping client_id to client information
            round_num: Current round number
        
        Returns:
            List of selected client IDs
        
        Raises:
            ValueError: If client_info is empty or invalid
        """
        if not client_info:
            raise ValueError("client_info cannot be empty")
        if not isinstance(client_info, dict):
            raise ValueError(f"client_info must be a dictionary, got {type(client_info)}")
        
        available_clients = list(client_info.keys())
        
        if not available_clients:
            raise ValueError("No available clients in client_info")
        
        num_to_select = min(self.clients_per_round, len(available_clients))
        
        if num_to_select == 0:
            logger.warning("Number of clients to select is 0")
            return []
        
        try:
            selected = self.rng.choice(
                available_clients,
                size=num_to_select,
                replace=False
            ).tolist()
        except Exception as e:
            logger.error(f"Error in random selection: {e}")
            raise
        
        self.update_selection_history(selected)
        return selected


class UniformSelector(FairClientSelector):
    """Uniform round-robin selection to ensure all clients participate equally."""
    
    def __init__(self, num_clients: int, clients_per_round: int, seed: int = 42):
        super().__init__(num_clients, clients_per_round, seed)
        self.client_queue = list(range(num_clients))
        self.rng.shuffle(self.client_queue)
        self.queue_position = 0
        
    def select_clients(
        self,
        client_info: Dict[int, Dict],
        round_num: int
    ) -> List[int]:
        """Select clients in round-robin fashion.
        
        Args:
            client_info: Dictionary mapping client_id to client information
            round_num: Current round number
        
        Returns:
            List of selected client IDs
        
        Raises:
            ValueError: If client_info is empty
        """
        if not client_info:
            raise ValueError("client_info cannot be empty")
        
        selected = []
        
        for _ in range(min(self.clients_per_round, len(self.client_queue))):
            if self.queue_position >= len(self.client_queue):
                # Reset queue with new shuffle
                self.rng.shuffle(self.client_queue)
                self.queue_position = 0
            
            client_id = self.client_queue[self.queue_position]
            
            # Only select if client is available
            if client_id in client_info:
                selected.append(client_id)
            
            self.queue_position += 1
        
        if not selected:
            logger.warning("No clients were selected in this round")
        
        self.update_selection_history(selected)
        return selected
