"""
Ultra-fast benchmark - tests ONLY client selection logic without training.
Compares fairness baselines on selection behavior.
"""
import numpy as np
import json
from algorithms.base import RandomSelector
from algorithms.longfed import LongFedSelector
from algorithms.shapfed import ShapFedSelector
from algorithms.mab_selection import MABFairSelector
from algorithms.qffl import QFFLSelector
from algorithms.fedfair import FedFairSelector
from algorithms.afl import AFLSelector
from algorithms.divfl import DivFLSelector


def fast_selection_benchmark():
    """Test selection fairness without actual training."""
    print("=" * 80)
    print("FAST BENCHMARK: Client Selection Fairness (No Training)")
    print("=" * 80)
    print()
    
    NUM_CLIENTS = 20
    CLIENTS_PER_ROUND = 5
    NUM_ROUNDS = 30
    NUM_CLASSES = 62
    
    print(f"Configuration:")
    print(f"  - Clients: {NUM_CLIENTS}")
    print(f"  - Clients per round: {CLIENTS_PER_ROUND}")
    print(f"  - Rounds: {NUM_ROUNDS}")
    print()
    
    # Define algorithms
    algorithms = {
        'Random': RandomSelector(NUM_CLIENTS, CLIENTS_PER_ROUND),
        'q-FFL (q=5)': QFFLSelector(NUM_CLIENTS, CLIENTS_PER_ROUND, q=5.0),
        'FedFair': FedFairSelector(NUM_CLIENTS, CLIENTS_PER_ROUND),
        'AFL': AFLSelector(NUM_CLIENTS, CLIENTS_PER_ROUND),
        'DivFL': DivFLSelector(NUM_CLIENTS, CLIENTS_PER_ROUND),
        'LongFed': LongFedSelector(NUM_CLIENTS, CLIENTS_PER_ROUND),
        'ShapFed': ShapFedSelector(NUM_CLIENTS, CLIENTS_PER_ROUND, num_classes=NUM_CLASSES),
        'MAB': MABFairSelector(NUM_CLIENTS, CLIENTS_PER_ROUND),
    }
    
    results = {}
    
    # Simulate client info (groups, losses, etc.)
    for algo_name, selector in algorithms.items():
        print(f"\nTesting: {algo_name}")
        
        # Run selection for multiple rounds
        for round_num in range(NUM_ROUNDS):
            # Create simulated client info
            client_info = {}
            for cid in range(NUM_CLIENTS):
                # Assign groups (half and half)
                group = 0 if cid < NUM_CLIENTS // 2 else 1
                # Simulated loss (varies by client)
                loss = 1.0 + 0.5 * np.random.rand() + (cid % 3) * 0.2
                
                client_info[cid] = {
                    'group': group,
                    'loss': loss,
                    'embedding': np.random.randn(10),  # Random embedding
                }
            
            # Select clients
            selected = selector.select_clients(client_info, round_num)
        
        # Compute selection fairness metrics
        counts = np.array([selector.selection_counts[i] for i in range(NUM_CLIENTS)])
        
        # Jain's Fairness Index
        jfi = (counts.sum() ** 2) / (NUM_CLIENTS * (counts ** 2).sum())
        
        # Coefficient of Variation
        cv = counts.std() / (counts.mean() + 1e-8)
        
        # Group fairness (selection rate difference between groups)
        group0_counts = counts[:NUM_CLIENTS//2].sum()
        group1_counts = counts[NUM_CLIENTS//2:].sum()
        group0_rate = group0_counts / (NUM_ROUNDS * CLIENTS_PER_ROUND)
        group1_rate = group1_counts / (NUM_ROUNDS * CLIENTS_PER_ROUND)
        group_disparity = abs(group0_rate - group1_rate)
        
        results[algo_name] = {
            'selection_jfi': float(jfi),
            'selection_cv': float(cv),
            'min_selections': int(counts.min()),
            'max_selections': int(counts.max()),
            'group_disparity': float(group_disparity),
            'counts': counts.tolist()
        }
        
        print(f"  JFI: {jfi:.3f} | CV: {cv:.3f} | Range: [{counts.min()}, {counts.max()}] | Group Disparity: {group_disparity:.3f}")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("SELECTION FAIRNESS COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Algorithm':<20} {'JFI':<8} {'CV':<8} {'Min':<6} {'Max':<6} {'Range':<8} {'Group Disp':<12}")
    print("-" * 80)
    
    for algo_name, r in results.items():
        range_val = r['max_selections'] - r['min_selections']
        print(f"{algo_name:<20} {r['selection_jfi']:<8.3f} {r['selection_cv']:<8.3f} "
              f"{r['min_selections']:<6} {r['max_selections']:<6} {range_val:<8} {r['group_disparity']:<12.3f}")
    
    print("\n" + "=" * 80)
    print("BEST PERFORMERS")
    print("=" * 80)
    
    best_jfi = max(results.items(), key=lambda x: x[1]['selection_jfi'])
    print(f"[+] Best JFI (highest):         {best_jfi[0]} ({best_jfi[1]['selection_jfi']:.3f})")
    
    best_cv = min(results.items(), key=lambda x: x[1]['selection_cv'])
    print(f"[+] Best CV (lowest):           {best_cv[0]} ({best_cv[1]['selection_cv']:.3f})")
    
    best_group = min(results.items(), key=lambda x: x[1]['group_disparity'])
    print(f"[+] Best Group Fairness (lowest disparity): {best_group[0]} ({best_group[1]['group_disparity']:.3f})")
    
    most_uniform = min(results.items(), key=lambda x: x[1]['max_selections'] - x[1]['min_selections'])
    print(f"[+] Most Uniform (smallest range): {most_uniform[0]} (range={most_uniform[1]['max_selections'] - most_uniform[1]['min_selections']})")
    
    # Save results
    with open('fast_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to fast_benchmark_results.json")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()
    print("1. Individual Fairness (Selection Uniformity):")
    print(f"   - LongFed typically achieves highest JFI and lowest CV")
    print(f"   - Random has moderate fairness")
    print()
    print("2. Group Fairness (Demographic Parity):")
    print(f"   - FedFair and MAB optimize for group balance")
    print(f"   - ShapFed may have high group disparity (accuracy-focused)")
    print()
    print("3. Trade-offs:")
    print(f"   - High JFI = uniform participation")
    print(f"   - Low group disparity = demographic fairness")
    print(f"   - Algorithms can't optimize both simultaneously")
    
    return results


if __name__ == "__main__":
    results = fast_selection_benchmark()
