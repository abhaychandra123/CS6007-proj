"""__init__ for utils package."""
from .model_utils import get_model, get_parameters, set_parameters, SimpleCNN, MLPClassifier
from .fairness_metrics import (
    statistical_parity_difference,
    equal_opportunity_difference,
    demographic_parity,
    compute_client_fairness_metrics,
    compute_fairness_score,
    compute_collaborative_fairness,
)
from .data_utils import (
    get_federated_cifar10,
    get_federated_adult,
    create_synthetic_adult_data,
)

__all__ = [
    'get_model',
    'get_parameters',
    'set_parameters',
    'SimpleCNN',
    'MLPClassifier',
    'statistical_parity_difference',
    'equal_opportunity_difference',
    'demographic_parity',
    'compute_client_fairness_metrics',
    'compute_fairness_score',
    'compute_collaborative_fairness',
    'get_federated_cifar10',
    'get_federated_adult',
    'create_synthetic_adult_data',
]
