"""
Demonstration of Fair Client Selection Algorithms.

This script demonstrates all three algorithms with a simple federated learning setup.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Import our algorithms
from algorithms import (
    LongFedSelector,
    ShapFedSelector,
    MABFairSelector,
    RandomSelector,
    compute_label_distribution
)

def demo_longfed():
    """Demonstrate LongFed algorithm."""
    print("\n" + "="*70)
    print("LONGFED DEMONSTRATION")
    print("="*70)
    
    num_clients = 10
    clients_per_round = 3
    num_rounds = 20
    
    # Initialize selector
    selector = LongFedSelector(
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        lambda_fair=0.5,
        seed=42
    )
    
    # Simulate client data distributions (label distributions)
    np.random.seed(42)
    num_classes = 10
    client_distributions = {}
    
    for i in range(num_clients):
        # Create heterogeneous distributions
        alpha = np.random.uniform(0.1, 2.0, num_classes)
        dist = np.random.dirichlet(alpha)
        client_distributions[i] = dist
    
    # Update selector with distributions
    selector.update_client_distributions(client_distributions)
    
    # Run selection for multiple rounds
    print(f"\nRunning {num_rounds} rounds with {clients_per_round} clients per round...")
    
    for round_num in range(num_rounds):
        # Create client_info dict
        client_info = {i: {'distribution': client_distributions[i]} 
                      for i in range(num_clients)}
        
        # Select clients
        selected = selector.select_clients(client_info, round_num)
        
        if round_num % 5 == 0:
            print(f"Round {round_num}: Selected clients {selected}")
    
    # Print fairness metrics
    print("\nFinal Fairness Metrics:")
    metrics = selector.get_fairness_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Plot selection distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    counts = [selector.selection_counts[i] for i in range(num_clients)]
    ax1.bar(range(num_clients), counts)
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Selection Count')
    ax1.set_title('LongFed: Client Selection Distribution')
    ax1.axhline(y=np.mean(counts), color='r', linestyle='--', label='Mean')
    ax1.legend()
    
    # Plot virtual queues
    queues = [selector.virtual_queues[i] for i in range(num_clients)]
    ax2.bar(range(num_clients), queues, color='orange')
    ax2.set_xlabel('Client ID')
    ax2.set_ylabel('Virtual Queue Value')
    ax2.set_title('LongFed: Virtual Queue State (Fairness Indicator)')
    
    plt.tight_layout()
    plt.savefig('demo_longfed.png', dpi=150)
    print("\nSaved visualization to demo_longfed.png")
    plt.close()


def demo_shapfed():
    """Demonstrate ShapFed algorithm."""
    print("\n" + "="*70)
    print("SHAPFED DEMONSTRATION")
    print("="*70)
    
    num_clients = 10
    clients_per_round = 3
    num_rounds = 20
    num_classes = 10
    
    # Initialize selector
    selector = ShapFedSelector(
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        num_classes=num_classes,
        selection_strategy='top_contribution',
        seed=42
    )
    
    # Simulate client contributions (will evolve over rounds)
    np.random.seed(42)
    feature_dim = 128
    
    print(f"\nRunning {num_rounds} rounds with contribution-based selection...")
    
    for round_num in range(num_rounds):
        # Simulate client parameters (last layer weights)
        client_params = {}
        global_params = torch.randn(num_classes, feature_dim)
        
        for i in range(num_clients):
            # Clients with higher quality have params closer to global
            quality = np.random.beta(2, 5)  # Skewed toward lower quality
            noise = torch.randn(num_classes, feature_dim) * (1 - quality)
            client_params[i] = global_params + noise
        
        # Update contributions
        selector.update_contributions(client_params, global_params)
        
        # Create client_info
        client_info = {i: {} for i in range(num_clients)}
        
        # Select clients
        selected = selector.select_clients(client_info, round_num)
        
        if round_num % 5 == 0:
            contributions = [selector.client_contributions[i] for i in selected]
            print(f"Round {round_num}: Selected {selected}, Contributions: {[f'{c:.3f}' for c in contributions]}")
    
    # Print fairness metrics
    print("\nFinal Fairness Metrics:")
    metrics = selector.get_fairness_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Plot contributions and selection
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    contributions = [selector.client_contributions[i] for i in range(num_clients)]
    counts = [selector.selection_counts[i] for i in range(num_clients)]
    
    # Contribution distribution
    ax1.bar(range(num_clients), contributions, color='green')
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Contribution Score')
    ax1.set_title('ShapFed: Client Contribution Scores')
    
    # Selection vs Contribution
    ax2.scatter(contributions, counts, s=100, alpha=0.6)
    ax2.set_xlabel('Contribution Score')
    ax2.set_ylabel('Selection Count')
    ax2.set_title('ShapFed: Selection vs Contribution (Collaborative Fairness)')
    
    # Add correlation
    if len(contributions) > 1:
        corr = np.corrcoef(contributions, counts)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax2.transAxes, fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('demo_shapfed.png', dpi=150)
    print("\nSaved visualization to demo_shapfed.png")
    plt.close()


def demo_mab():
    """Demonstrate MAB Fair Selection algorithm."""
    print("\n" + "="*70)
    print("MAB FAIR SELECTION DEMONSTRATION")
    print("="*70)
    
    num_clients = 10
    clients_per_round = 3
    num_rounds = 20
    
    # Initialize selector
    selector = MABFairSelector(
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        epsilon=0.2,
        alpha_fair=1.0,
        beta_fair=1.0,
        gamma_acc=0.5,
        seed=42
    )
    
    # Simulate client fairness metrics and accuracies
    np.random.seed(42)
    
    # Some clients are more fair than others
    base_fairness = {i: np.random.beta(2, 2) for i in range(num_clients)}
    
    print(f"\nRunning {num_rounds} rounds with epsilon-greedy selection (ε={selector.epsilon})...")
    
    global_fairness = 0.5  # Start with moderate unfairness
    global_acc = 0.5
    
    for round_num in range(num_rounds):
        # Create client_info with fairness
        client_fairness = {i: base_fairness[i] + np.random.normal(0, 0.05) 
                          for i in range(num_clients)}
        client_fairness = {i: np.clip(f, 0, 1) for i, f in client_fairness.items()}
        
        client_info = {i: {'fairness': client_fairness[i]} 
                      for i in range(num_clients)}
        
        # Select clients
        selected = selector.select_clients(client_info, round_num)
        
        # Simulate outcomes
        old_fairness = global_fairness
        old_acc = global_acc
        
        # Global fairness improves if we select fair clients
        avg_selected_fairness = np.mean([client_fairness[i] for i in selected])
        global_fairness = 0.9 * global_fairness + 0.1 * avg_selected_fairness
        
        # Accuracy improves gradually
        global_acc = min(0.95, global_acc + np.random.normal(0.01, 0.005))
        
        # Update rewards
        selector.update_rewards(
            selected,
            global_fairness,
            old_fairness,
            global_acc,
            old_acc,
            client_fairness
        )
        
        if round_num % 5 == 0:
            print(f"Round {round_num}: Selected {selected}, Global Fairness: {global_fairness:.3f}, Acc: {global_acc:.3f}")
    
    # Print fairness metrics
    print("\nFinal Fairness Metrics:")
    metrics = selector.get_fairness_metrics()
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        print(f"  {key}: {value:.4f}")
    
    # Plot rewards and fairness evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean rewards
    rewards = [selector.mean_rewards[i] for i in range(num_clients)]
    counts = [selector.selection_counts[i] for i in range(num_clients)]
    
    ax1.bar(range(num_clients), rewards, color='purple')
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('MAB: Client Mean Rewards')
    
    # Fairness evolution
    if selector.global_fairness_history:
        ax2.plot(selector.global_fairness_history, linewidth=2, marker='o')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Global Fairness Metric (SPD)')
        ax2.set_title('MAB: Global Fairness Evolution')
        ax2.axhline(y=0.1, color='r', linestyle='--', label='Target')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_mab.png', dpi=150)
    print("\nSaved visualization to demo_mab.png")
    plt.close()


def comparison():
    """Compare all algorithms."""
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON")
    print("="*70)
    
    num_clients = 10
    clients_per_round = 3
    num_rounds = 30
    
    algorithms = {
        'Random': RandomSelector(num_clients, clients_per_round, seed=42),
        'LongFed': LongFedSelector(num_clients, clients_per_round, lambda_fair=0.5, seed=42),
        'ShapFed': ShapFedSelector(num_clients, clients_per_round, num_classes=10, seed=42),
        'MAB': MABFairSelector(num_clients, clients_per_round, epsilon=0.2, seed=42),
    }
    
    # Initialize distributions for LongFed
    np.random.seed(42)
    num_classes = 10
    client_distributions = {i: np.random.dirichlet(np.ones(num_classes)) 
                           for i in range(num_clients)}
    algorithms['LongFed'].update_client_distributions(client_distributions)
    
    print(f"\nRunning {num_rounds} rounds for each algorithm...")
    
    results = defaultdict(lambda: {'counts': {i: 0 for i in range(num_clients)}})
    
    for alg_name, selector in algorithms.items():
        for round_num in range(num_rounds):
            client_info = {i: {'distribution': client_distributions[i]} 
                          for i in range(num_clients)}
            
            selected = selector.select_clients(client_info, round_num)
            
            for cid in selected:
                results[alg_name]['counts'][cid] += 1
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (alg_name, selector) in enumerate(algorithms.items()):
        counts = [results[alg_name]['counts'][i] for i in range(num_clients)]
        
        axes[idx].bar(range(num_clients), counts)
        axes[idx].set_xlabel('Client ID')
        axes[idx].set_ylabel('Selection Count')
        axes[idx].set_title(f'{alg_name} Selection Distribution')
        axes[idx].axhline(y=np.mean(counts), color='r', linestyle='--', label='Mean')
        
        # Add CV
        cv = np.std(counts) / (np.mean(counts) + 1e-8)
        axes[idx].text(0.95, 0.95, f'CV: {cv:.3f}', 
                      transform=axes[idx].transAxes, fontsize=10,
                      verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('demo_comparison.png', dpi=150)
    print("\nSaved comparison visualization to demo_comparison.png")
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"{'Algorithm':<15} {'Mean':<10} {'Std':<10} {'CV':<10} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    
    for alg_name in algorithms.keys():
        counts = np.array([results[alg_name]['counts'][i] for i in range(num_clients)])
        print(f"{alg_name:<15} {counts.mean():<10.2f} {counts.std():<10.2f} "
              f"{counts.std()/(counts.mean()+1e-8):<10.3f} {counts.min():<10} {counts.max():<10}")


def main():
    """Run all demonstrations."""
    print("="*70)
    print("FAIR CLIENT SELECTION - DEMONSTRATION")
    print("="*70)
    print("\nThis script demonstrates three seminal algorithms:")
    print("1. LongFed - Individual fairness via Lyapunov optimization")
    print("2. ShapFed - Collaborative fairness via Shapley values")
    print("3. MAB - Demographic fairness via multi-armed bandits")
    print("\nGenerating demonstrations...")
    
    try:
        demo_longfed()
        demo_shapfed()
        demo_mab()
        comparison()
        
        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  - demo_longfed.png: LongFed selection and virtual queues")
        print("  - demo_shapfed.png: ShapFed contributions and collaborative fairness")
        print("  - demo_mab.png: MAB rewards and fairness evolution")
        print("  - demo_comparison.png: Side-by-side comparison of all algorithms")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
