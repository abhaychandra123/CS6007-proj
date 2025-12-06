"""Fairness metrics computation for federated learning."""
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)


def statistical_parity_difference(y_true, y_pred, sensitive_attr):
    """
    Compute Statistical Parity Difference (SPD).
    
    SPD = P(Y_pred=1|A=0) - P(Y_pred=1|A=1)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute (0 or 1)
    
    Returns:
        SPD value (0 means perfect fairness)
    
    Raises:
        ValueError: If inputs are empty or have mismatched lengths
    """
    # Input validation
    if len(y_pred) == 0 or len(sensitive_attr) == 0:
        raise ValueError("y_pred and sensitive_attr cannot be empty")
    
    y_pred = np.array(y_pred)
    sensitive_attr = np.array(sensitive_attr)
    
    if len(y_pred) != len(sensitive_attr):
        raise ValueError(f"Length mismatch: y_pred={len(y_pred)}, sensitive_attr={len(sensitive_attr)}")
    
    # Validate sensitive attribute values
    unique_attrs = np.unique(sensitive_attr)
    if len(unique_attrs) == 0:
        logger.warning("No unique sensitive attributes found")
        return 0.0
    
    # Group 0 (unprivileged)
    group_0_mask = sensitive_attr == 0
    if group_0_mask.sum() > 0:
        p_y1_a0 = (y_pred[group_0_mask] == 1).mean()
    else:
        logger.warning("No samples in group 0")
        p_y1_a0 = 0.0
    
    # Group 1 (privileged)
    group_1_mask = sensitive_attr == 1
    if group_1_mask.sum() > 0:
        p_y1_a1 = (y_pred[group_1_mask] == 1).mean()
    else:
        logger.warning("No samples in group 1")
        p_y1_a1 = 0.0
    
    spd = abs(p_y1_a0 - p_y1_a1)
    return float(spd)


def equal_opportunity_difference(y_true, y_pred, sensitive_attr):
    """
    Compute Equal Opportunity Difference (EOD).
    
    EOD = |P(Y_pred=1|Y=1,A=0) - P(Y_pred=1|Y=1,A=1)|
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute (0 or 1)
    
    Returns:
        EOD value (0 means perfect fairness)
    
    Raises:
        ValueError: If inputs are empty or have mismatched lengths
    """
    # Input validation
    if len(y_true) == 0 or len(y_pred) == 0 or len(sensitive_attr) == 0:
        raise ValueError("Inputs cannot be empty")
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive_attr = np.array(sensitive_attr)
    
    if not (len(y_true) == len(y_pred) == len(sensitive_attr)):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, sensitive_attr={len(sensitive_attr)}")
    
    # True positive rate for group 0
    group_0_pos = (sensitive_attr == 0) & (y_true == 1)
    if group_0_pos.sum() > 0:
        tpr_0 = (y_pred[group_0_pos] == 1).mean()
    else:
        logger.warning("No positive samples in group 0")
        tpr_0 = 0.0
    
    # True positive rate for group 1
    group_1_pos = (sensitive_attr == 1) & (y_true == 1)
    if group_1_pos.sum() > 0:
        tpr_1 = (y_pred[group_1_pos] == 1).mean()
    else:
        logger.warning("No positive samples in group 1")
        tpr_1 = 0.0
    
    eod = abs(tpr_0 - tpr_1)
    return float(eod)


def demographic_parity(y_pred, sensitive_attr):
    """
    Compute Demographic Parity metric.
    
    Returns ratio of positive predictions across groups.
    """
    y_pred = np.array(y_pred)
    sensitive_attr = np.array(sensitive_attr)
    
    group_0_mask = sensitive_attr == 0
    group_1_mask = sensitive_attr == 1
    
    if group_0_mask.sum() > 0 and group_1_mask.sum() > 0:
        rate_0 = (y_pred[group_0_mask] == 1).mean()
        rate_1 = (y_pred[group_1_mask] == 1).mean()
        return min(rate_0, rate_1) / max(rate_0, rate_1) if max(rate_0, rate_1) > 0 else 1.0
    return 1.0


def compute_client_fairness_metrics(y_true, y_pred, sensitive_attr) -> Dict[str, float]:
    """
    Compute comprehensive fairness metrics for a client.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attr: Sensitive attribute values
    
    Returns:
        Dictionary of fairness metrics
    """
    metrics = {
        'spd': statistical_parity_difference(y_true, y_pred, sensitive_attr),
        'eod': equal_opportunity_difference(y_true, y_pred, sensitive_attr),
        'dp': demographic_parity(y_pred, sensitive_attr),
    }
    
    # Compute accuracy per group
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive_attr = np.array(sensitive_attr)
    
    for group in [0, 1]:
        mask = sensitive_attr == group
        if mask.sum() > 0:
            acc = (y_true[mask] == y_pred[mask]).mean()
            metrics[f'acc_group_{group}'] = acc
    
    return metrics


def compute_fairness_score(metrics: Dict[str, float]) -> float:
    """
    Compute overall fairness score.
    
    Lower SPD and EOD are better. Returns score in [0, 1] where 1 is perfectly fair.
    """
    spd = metrics.get('spd', 1.0)
    eod = metrics.get('eod', 1.0)
    
    # Fairness score: closer to 1 is more fair
    fairness = 1.0 / (1.0 + spd + eod)
    return fairness


def compute_contribution_variance(client_contributions: List[float]) -> float:
    """
    Compute variance in client contributions (lower is more fair).
    """
    return np.var(client_contributions)


def compute_selection_fairness(selection_counts: Dict[int, int], num_rounds: int) -> Dict[str, float]:
    """
    Compute fairness of client selection over rounds.
    
    Args:
        selection_counts: Dictionary mapping client_id to selection count
        num_rounds: Total number of rounds
    
    Returns:
        Dictionary with selection fairness metrics
    
    Raises:
        ValueError: If selection_counts is empty or num_rounds is invalid
    """
    if not selection_counts:
        raise ValueError("selection_counts cannot be empty")
    if num_rounds <= 0:
        raise ValueError(f"num_rounds must be positive, got {num_rounds}")
    
    counts = np.array(list(selection_counts.values()))
    total_clients = len(selection_counts)
    
    if total_clients == 0:
        raise ValueError("No clients in selection_counts")
    
    # Expected count if perfectly uniform
    expected_count = num_rounds * (1.0 / total_clients)
    
    mean_count = counts.mean()
    std_count = counts.std()
    
    # Avoid division by zero
    cv = std_count / mean_count if mean_count > 1e-10 else 0.0
    
    return {
        'selection_variance': float(np.var(counts)),
        'selection_std': float(std_count),
        'selection_min': int(np.min(counts)),
        'selection_max': int(np.max(counts)),
        'selection_cv': float(cv),
        'expected_count': float(expected_count),
    }


def compute_performance_variance(client_accuracies: List[float]) -> float:
    """
    Compute variance in client performance (lower is more fair).
    """
    return np.var(client_accuracies)


def compute_collaborative_fairness(
    client_accuracies: List[float],
    client_contributions: List[float]
) -> float:
    """
    Compute collaborative fairness: correlation between contribution and accuracy.
    
    Higher correlation means better collaborative fairness (high contributors
    get better models).
    
    Args:
        client_accuracies: List of client accuracy values
        client_contributions: List of client contribution values
    
    Returns:
        Pearson correlation coefficient, or 0.0 if cannot be computed
    
    Raises:
        ValueError: If inputs have mismatched lengths
    """
    if len(client_accuracies) != len(client_contributions):
        raise ValueError(f"Length mismatch: accuracies={len(client_accuracies)}, contributions={len(client_contributions)}")
    
    if len(client_accuracies) < 2:
        logger.warning("Need at least 2 clients to compute correlation")
        return 0.0
    
    # Check for zero variance
    acc_var = np.var(client_accuracies)
    cont_var = np.var(client_contributions)
    
    if acc_var < 1e-10 or cont_var < 1e-10:
        logger.warning("Zero variance in inputs, cannot compute correlation")
        return 0.0
    
    try:
        corr_matrix = np.corrcoef(client_contributions, client_accuracies)
        return float(corr_matrix[0, 1])
    except Exception as e:
        logger.error(f"Error computing correlation: {e}")
        return 0.0
