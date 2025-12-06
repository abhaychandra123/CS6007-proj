"""
Quick benchmark comparing all fair client selection algorithms.
Tests on FEMNIST with reduced parameters for speed.
"""
import sys
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from test_femnist import FederatedTrainer, run_experiment
from algorithms.base import RandomSelector
from algorithms.longfed import LongFedSelector
from algorithms.shapfed import ShapFedSelector
from algorithms.mab_selection import MABFairSelector
from algorithms.qffl import QFFLSelector
from algorithms.fedfair import FedFairSelector
from algorithms.afl import AFLSelector
from algorithms.divfl import DivFLSelector


def create_femnist_like_data(num_clients=20, samples_per_client=300, num_classes=62):
    """
    Create FEMNIST-like synthetic federated data locally (no network required).
    
    Simulates FEMNIST characteristics:
    - 62 classes (10 digits + 26 uppercase + 26 lowercase)
    - Non-IID: Each writer prefers certain character categories
    - Variable samples per client
    - Grayscale 28x28 images
    """
    print(f"\nCreating FEMNIST-like data...")
    print(f"  - {num_clients} clients")
    print(f"  - ~{samples_per_client} samples per client")
    print(f"  - {num_classes} classes (digits + letters)")
    
    client_datasets = {}
    all_test_X = []
    all_test_y = []
    
    # Create client datasets with non-IID distribution
    for client_id in range(num_clients):
        # Determine client's preferred class range (simulating writer preferences)
        if client_id < num_clients // 3:
            # First 1/3 of clients prefer digits (0-9)
            preferred_classes = list(range(10))
        elif client_id < 2 * num_clients // 3:
            # Middle 1/3 prefer uppercase (10-35)
            preferred_classes = list(range(10, 36))
        else:
            # Last 1/3 prefer lowercase (36-61)
            preferred_classes = list(range(36, 62))
        
        # Generate samples (80% from preferred, 20% random)
        num_samples = samples_per_client
        X_list = []
        y_list = []
        
        for _ in range(num_samples):
            if np.random.rand() < 0.8:
                # Preferred class
                y = np.random.choice(preferred_classes)
            else:
                # Random class
                y = np.random.randint(0, num_classes)
            
            # Generate realistic-looking handwritten character (random noise + structure)
            img = np.random.randn(28, 28) * 0.1 + 0.1
            # Add some structure based on class
            img[10:18, 10:18] += (y / num_classes) * 0.5
            
            X_list.append(img)
            y_list.append(y)
        
        X = torch.FloatTensor(X_list).unsqueeze(1)  # Add channel dimension
        y = torch.LongTensor(y_list)
        
        client_datasets[client_id] = (X, y)  # Store as tuple
        
        # Add some samples to test set (10% of client data)
        test_size = num_samples // 10
        all_test_X.append(X[:test_size])
        all_test_y.append(y[:test_size])
    
    # Create test dataset
    test_X = torch.cat(all_test_X, dim=0)
    test_y = torch.cat(all_test_y, dim=0)
    
    # Create sensitive attributes (groups)
    # Group 0: first half of clients, Group 1: second half
    client_sensitive = {}
    for client_id in range(num_clients):
        group = 0 if client_id < num_clients // 2 else 1
        X, y = client_datasets[client_id]
        num_samples = len(y)
        client_sensitive[client_id] = torch.full((num_samples,), group, dtype=torch.long)
    
    # Test sensitive attributes (alternating for diversity)
    test_sensitive = torch.tensor([i % 2 for i in range(len(test_y))], dtype=torch.long)
    
    print(f"  [OK] Created {num_clients} clients with {sum(len(ds[1]) for ds in client_datasets.values())} training samples")
    print(f"  [OK] Test set: {len(test_X)} samples")
    
    return client_datasets, (test_X, test_y), client_sensitive, test_sensitive


def quick_benchmark():
    """Run quick benchmark of all algorithms."""
    print("=" * 80)
    print("QUICK BENCHMARK: Fair Client Selection Algorithms")
    print("=" * 80)
    print()
    
    # Reduced parameters for speed
    NUM_CLIENTS = 10
    CLIENTS_PER_ROUND = 3
    NUM_ROUNDS = 10  # Reduced from 15
    NUM_CLASSES = 62
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Configuration:")
    print(f"  - Clients: {NUM_CLIENTS}")
    print(f"  - Clients per round: {CLIENTS_PER_ROUND}")
    print(f"  - Training rounds: {NUM_ROUNDS}")
    print(f"  - Classes: {NUM_CLASSES}")
    print(f"  - Device: {DEVICE}")
    print()
    
    # Create FEMNIST-like data once
    print("Creating FEMNIST-like dataset...")
    client_datasets, test_dataset, client_sensitive, test_sensitive = create_femnist_like_data(
        num_clients=NUM_CLIENTS,
        samples_per_client=200,  # Reduced from 300
        num_classes=NUM_CLASSES
    )
    test_X, test_y = test_dataset
    print()
    
    # Define all algorithms to test
    algorithms = {
        'Random': (RandomSelector, {}),
        'LongFed': (LongFedSelector, {}),  # Uses default parameters
        'ShapFed': (ShapFedSelector, {'num_classes': NUM_CLASSES}),
        'MAB': (MABFairSelector, {}),  # Uses default parameters
        'q-FFL': (QFFLSelector, {'q': 5.0}),
        'FedFair': (FedFairSelector, {}),
        'AFL': (AFLSelector, {'learning_rate': 0.01}),
        'DivFL': (DivFLSelector, {'diversity_weight': 0.5}),
    }
    
    results = {}
    
    # Run each algorithm
    for algo_name, (selector_class, params) in algorithms.items():
        print(f"\n{'=' * 80}")
        print(f"Testing: {algo_name}")
        print(f"{'=' * 80}")
        
        start_time = time.time()
        
        try:
            # Create trainer
            trainer = FederatedTrainer(
                client_datasets=client_datasets,
                test_dataset=(test_X, test_y),
                client_sensitive=client_sensitive,
                test_sensitive=test_sensitive,
                num_classes=NUM_CLASSES,
                device=DEVICE
            )
            
            # Create selector
            selector = selector_class(
                num_clients=NUM_CLIENTS,
                clients_per_round=CLIENTS_PER_ROUND,
                **params
            )
            
            # Run experiment
            result = run_experiment(
                trainer=trainer,
                selector=selector,
                num_rounds=NUM_ROUNDS,
                algorithm_name=algo_name
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract final metrics - use lists 'accuracy', 'spd', 'eod'
            final_acc = result['accuracy'][-1] if result['accuracy'] else 0.0
            final_spd = result['spd'][-1] if result['spd'] else 0.0
            final_eod = result['eod'][-1] if result['eod'] else 0.0
            
            # Selection fairness
            selection_counts = list(result['client_selection_counts'].values())
            selection_cv = np.std(selection_counts) / (np.mean(selection_counts) + 1e-8)
            selection_jfi = (sum(selection_counts) ** 2) / (len(selection_counts) * sum(c**2 for c in selection_counts))
            
            results[algo_name] = {
                'accuracy': final_acc,
                'spd': final_spd,
                'eod': final_eod,
                'selection_cv': selection_cv,
                'selection_jfi': selection_jfi,
                'time_seconds': elapsed_time,
                'selection_counts': result['client_selection_counts']
            }
            
            print(f"\n[OK] {algo_name} completed in {elapsed_time:.1f}s")
            print(f"  - Accuracy: {final_acc:.3f}")
            print(f"  - SPD: {final_spd:.3f}")
            print(f"  - EOD: {final_eod:.3f}")
            print(f"  - Selection CV: {selection_cv:.3f}")
            print(f"  - Selection JFI: {selection_jfi:.3f}")
            
        except Exception as e:
            print(f"\n[X] {algo_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results[algo_name] = {'error': str(e)}
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()
    
    print(f"{'Algorithm':<15} {'Accuracy':<10} {'SPD':<10} {'EOD':<10} {'Sel.CV':<10} {'Sel.JFI':<10} {'Time(s)':<10}")
    print("-" * 95)
    
    for algo_name in algorithms.keys():
        if algo_name in results and 'error' not in results[algo_name]:
            r = results[algo_name]
            print(f"{algo_name:<15} {r['accuracy']:<10.3f} {r['spd']:<10.3f} {r['eod']:<10.3f} "
                  f"{r['selection_cv']:<10.3f} {r['selection_jfi']:<10.3f} {r['time_seconds']:<10.1f}")
        else:
            print(f"{algo_name:<15} {'FAILED'}")
    
    print()
    
    # Identify best performers
    print("\n" + "=" * 80)
    print("BEST PERFORMERS BY METRIC")
    print("=" * 80)
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        # Best accuracy
        best_acc = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"[+] Best Accuracy: {best_acc[0]} ({best_acc[1]['accuracy']:.3f})")
        
        # Best SPD (lowest)
        best_spd = min(valid_results.items(), key=lambda x: x[1]['spd'])
        print(f"[+] Best SPD (lowest): {best_spd[0]} ({best_spd[1]['spd']:.3f})")
        
        # Best EOD (lowest)
        best_eod = min(valid_results.items(), key=lambda x: x[1]['eod'])
        print(f"[+] Best EOD (lowest): {best_eod[0]} ({best_eod[1]['eod']:.3f})")
        
        # Best Selection CV (lowest)
        best_cv = min(valid_results.items(), key=lambda x: x[1]['selection_cv'])
        print(f"[+] Best Selection CV (lowest): {best_cv[0]} ({best_cv[1]['selection_cv']:.3f})")
        
        # Best Selection JFI (highest)
        best_jfi = max(valid_results.items(), key=lambda x: x[1]['selection_jfi'])
        print(f"[+] Best Selection JFI (highest): {best_jfi[0]} ({best_jfi[1]['selection_jfi']:.3f})")
        
        # Fastest
        fastest = min(valid_results.items(), key=lambda x: x[1]['time_seconds'])
        print(f"[+] Fastest: {fastest[0]} ({fastest[1]['time_seconds']:.1f}s)")
    
    # Save results
    output_file = 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = quick_benchmark()
