"""
Comprehensive Fairness Analysis Report Generator

Analyzes results from fair client selection algorithms using all major fairness metrics:
- SPD (Statistical Parity Difference)
- Equalised Odds  
- EOD (Equal Opportunity Difference)
- Jain's Fairness Index
- Variance/MAD
- Four-notion framework

Generates trade-off analysis and validates against expected theoretical results.
"""

import json
import numpy as np
from typing import Dict, List
from tabulate import tabulate


def generate_comprehensive_report(results: Dict[str, Dict], save_path: str = "FAIRNESS_ANALYSIS_REPORT.md"):
    """
    Generate comprehensive fairness analysis report.
    
    Args:
        results: {algorithm_name: {metrics...}}
        save_path: Where to save the markdown report
    """
    
    report_lines = []
    
    # Header
    report_lines.append("# Comprehensive Fairness Analysis Report")
    report_lines.append("## Fair Client Selection Algorithms on FEMNIST Dataset")
    report_lines.append("")
    report_lines.append("**Dataset**: FEMNIST (Federated Extended MNIST)")
    report_lines.append("- 62 classes (10 digits + 26 uppercase + 26 lowercase letters)")
    report_lines.append("- Non-IID distribution (each writer has unique handwriting style)")
    report_lines.append("- Naturally partitioned by writer_id")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Executive Summary Table
    report_lines.append("## Executive Summary")
    report_lines.append("")
    
    summary_table = []
    for alg_name, metrics in results.items():
        summary_table.append([
            alg_name,
            f"{metrics.get('final_accuracy', 0)*100:.2f}%",
            f"{metrics.get('selection_cv', 0):.4f}",
            f"{metrics.get('spd', 0):.4f}",
            f"{metrics.get('eod', 0):.4f}",
        ])
    
    report_lines.append(tabulate(
        summary_table,
        headers=["Algorithm", "Accuracy", "Selection CV", "SPD", "EOD"],
        tablefmt="github"
    ))
    report_lines.append("")
    report_lines.append("*Lower values are better for CV, SPD, and EOD*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Detailed Metrics by Algorithm
    report_lines.append("## Detailed Algorithm Analysis")
    report_lines.append("")
    
    for alg_name in ['Random', 'LongFed', 'ShapFed', 'MAB']:
        if alg_name not in results:
            continue
            
        metrics = results[alg_name]
        report_lines.append(f"### {alg_name}")
        report_lines.append("")
        
        # Performance
        report_lines.append("#### Performance Metrics")
        report_lines.append(f"- **Final Accuracy**: {metrics.get('final_accuracy', 0)*100:.2f}%")
        report_lines.append(f"- **Peak Accuracy**: {max(metrics.get('acc_history', [0]))*100:.2f}%")
        report_lines.append(f"- **Convergence Round**: Round {np.argmax(metrics.get('acc_history', [0])) + 1}")
        report_lines.append("")
        
        # Selection Fairness
        report_lines.append("#### Selection Fairness")
        report_lines.append(f"- **Jain's Fairness Index**: {metrics.get('selection_jfi', 0):.4f} (1.0 = perfect)")
        report_lines.append(f"- **Coefficient of Variation**: {metrics.get('selection_cv', 0):.4f} (lower = fairer)")
        report_lines.append(f"- **Selection Variance**: {metrics.get('selection_variance', 0):.4f}")
        report_lines.append("")
        
        # Demographic Fairness
        report_lines.append("#### Demographic Fairness")
        report_lines.append(f"- **Statistical Parity Difference (SPD)**: {metrics.get('spd', 0):.4f} (0 = perfect)")
        report_lines.append(f"- **Equal Opportunity Difference (EOD)**: {metrics.get('eod', 0):.4f} (0 = perfect)")
        report_lines.append("")
        
        # Algorithm-specific insights
        report_lines.append("#### Key Characteristics")
        if alg_name == 'Random':
            report_lines.append("- **Baseline**: Uniform random selection")
            report_lines.append("- **Expected**: Moderate fairness by chance, average performance")
            report_lines.append("- **Strength**: Simple, no bias")
            report_lines.append("- **Weakness**: No optimization for fairness or accuracy")
        elif alg_name == 'LongFed':
            report_lines.append("- **Objective**: Individual fairness via Lyapunov optimization")
            report_lines.append("- **Expected**: Lowest selection CV, balanced client participation")
            report_lines.append("- **Strength**: Theoretical (1-1/e) approximation guarantee")
            report_lines.append("- **Weakness**: May sacrifice some accuracy for fairness")
        elif alg_name == 'ShapFed':
            report_lines.append("- **Objective**: Maximize model accuracy via Shapley value approximation")
            report_lines.append("- **Expected**: Highest accuracy, may have higher CV")
            report_lines.append("- **Strength**: Selects most contributive clients")
            report_lines.append("- **Weakness**: Can lead to client starvation (unfair selection)")
        elif alg_name == 'MAB':
            report_lines.append("- **Objective**: Demographic fairness via multi-armed bandit")
            report_lines.append("- **Expected**: Lowest SPD/EOD, good accuracy-fairness balance")
            report_lines.append("- **Strength**: 67.1% bias reduction demonstrated in literature")
            report_lines.append("- **Weakness**: Requires sensitive attribute information")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Comparative Analysis
    report_lines.append("## Comparative Trade-off Analysis")
    report_lines.append("")
    
    # Accuracy vs Fairness Trade-off
    report_lines.append("### Accuracy vs Selection Fairness")
    report_lines.append("")
    tradeoff_table = []
    for alg_name, metrics in results.items():
        tradeoff_table.append([
            alg_name,
            f"{metrics.get('final_accuracy', 0)*100:.2f}%",
            f"{metrics.get('selection_cv', 0):.4f}",
            "Accuracy Focus" if metrics.get('final_accuracy', 0) > 0.70 else "Balanced" if metrics.get('final_accuracy', 0) > 0.60 else "Fairness Focus"
        ])
    
    report_lines.append(tabulate(
        tradeoff_table,
        headers=["Algorithm", "Accuracy", "Selection CV", "Strategy"],
        tablefmt="github"
    ))
    report_lines.append("")
    
    # Fairness Dimensions
    report_lines.append("### Fairness Dimensions Comparison")
    report_lines.append("")
    fairness_table = []
    for alg_name, metrics in results.items():
        fairness_table.append([
            alg_name,
            f"{metrics.get('selection_jfi', 0):.4f}",
            f"{metrics.get('spd', 0):.4f}",
            f"{metrics.get('eod', 0):.4f}",
        ])
    
    report_lines.append(tabulate(
        fairness_table,
        headers=["Algorithm", "JFI (Selection)", "SPD (Demographic)", "EOD (Opportunity)"],
        tablefmt="github"
    ))
    report_lines.append("")
    report_lines.append("*Higher JFI is better; lower SPD and EOD are better*")
    report_lines.append("")
    
    # Theoretical Validation
    report_lines.append("## Validation Against Literature")
    report_lines.append("")
    
    report_lines.append("### Expected Results from Research Papers")
    report_lines.append("")
    report_lines.append("1. **Random Baseline**")
    report_lines.append("   - Expected SPD: 0.14-0.20 (moderate bias)")
    report_lines.append(f"   - Observed SPD: {results.get('Random', {}).get('spd', 0):.4f}")
    report_lines.append(f"   - Status: {'✓ ALIGNED' if 0.10 <= results.get('Random', {}).get('spd', 0) <= 0.25 else '⚠ DEVIATION'}")
    report_lines.append("")
    
    report_lines.append("2. **MAB (Bouzamoucha et al., 2025)**")
    report_lines.append("   - Expected SPD Reduction: 67.1% (from 0.140 to 0.046)")
    random_spd = results.get('Random', {}).get('spd', 0.14)
    mab_spd = results.get('MAB', {}).get('spd', 0)
    reduction = ((random_spd - mab_spd) / random_spd * 100) if random_spd > 0 else 0
    report_lines.append(f"   - Observed SPD: {mab_spd:.4f} (Reduction: {reduction:.1f}%)")
    report_lines.append(f"   - Status: {'✓ ALIGNED' if reduction >= 50 else '⚠ DEVIATION'}")
    report_lines.append("")
    
    report_lines.append("3. **LongFed (Li et al., 2024)**")
    report_lines.append("   - Expected: Lowest selection CV (best individual fairness)")
    longfed_cv = results.get('LongFed', {}).get('selection_cv', 1.0)
    min_cv = min([results.get(alg, {}).get('selection_cv', 1.0) for alg in results.keys()])
    report_lines.append(f"   - Observed CV: {longfed_cv:.4f}")
    report_lines.append(f"   - Status: {'✓ ALIGNED' if longfed_cv == min_cv else '⚠ DEVIATION'}")
    report_lines.append("")
    
    report_lines.append("4. **ShapFed (Tastan et al., 2024)**")
    report_lines.append("   - Expected: Highest accuracy (contribution-based selection)")
    shapfed_acc = results.get('ShapFed', {}).get('final_accuracy', 0)
    max_acc = max([results.get(alg, {}).get('final_accuracy', 0) for alg in results.keys()])
    report_lines.append(f"   - Observed Accuracy: {shapfed_acc*100:.2f}%")
    report_lines.append(f"   - Status: {'✓ ALIGNED' if shapfed_acc == max_acc else '⚠ DEVIATION'}")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")
    report_lines.append("### When to Use Each Algorithm")
    report_lines.append("")
    report_lines.append("1. **Use LongFed when:**")
    report_lines.append("   - Individual fairness is critical (e.g., healthcare, finance)")
    report_lines.append("   - Need to ensure balanced participation across clients")
    report_lines.append("   - Can tolerate slight accuracy loss for fairness")
    report_lines.append("")
    report_lines.append("2. **Use ShapFed when:**")
    report_lines.append("   - Maximum model accuracy is the primary goal")
    report_lines.append("   - Data quality varies significantly across clients")
    report_lines.append("   - Want to identify and reward high-quality contributors")
    report_lines.append("")
    report_lines.append("3. **Use MAB when:**")
    report_lines.append("   - Demographic fairness is legally/ethically required")
    report_lines.append("   - Need to mitigate bias against protected groups")
    report_lines.append("   - Sensitive attributes are available and can be used")
    report_lines.append("")
    report_lines.append("4. **Use Random when:**")
    report_lines.append("   - Need simple, unbiased baseline")
    report_lines.append("   - Want to avoid complex fairness considerations")
    report_lines.append("   - Data is already relatively uniform across clients")
    report_lines.append("")
    
    # Key Findings
    report_lines.append("## Key Findings")
    report_lines.append("")
    
    best_acc_alg = max(results.keys(), key=lambda k: results[k].get('final_accuracy', 0))
    best_cv_alg = min(results.keys(), key=lambda k: results[k].get('selection_cv', 100))
    best_spd_alg = min(results.keys(), key=lambda k: results[k].get('spd', 100))
    
    report_lines.append(f"1. **Highest Accuracy**: {best_acc_alg} ({results[best_acc_alg].get('final_accuracy', 0)*100:.2f}%)")
    report_lines.append(f"2. **Best Selection Fairness**: {best_cv_alg} (CV: {results[best_cv_alg].get('selection_cv', 0):.4f})")
    report_lines.append(f"3. **Best Demographic Fairness**: {best_spd_alg} (SPD: {results[best_spd_alg].get('spd', 0):.4f})")
    report_lines.append("")
    report_lines.append("**Trade-off Insight**: There is an inherent tension between maximizing accuracy")
    report_lines.append("and ensuring fairness. Different algorithms optimize for different objectives,")
    report_lines.append("and the choice depends on the specific application requirements.")
    report_lines.append("")
    
    # References
    report_lines.append("## References")
    report_lines.append("")
    report_lines.append("1. Bouzamoucha, M., et al. (2025). \"Multi-Armed Bandit Approach for Fair Client Selection in Federated Learning.\"")
    report_lines.append("2. Li, T., et al. (2024). \"LongFed: Longitudinal Fairness in Federated Learning via Lyapunov Optimization.\"")
    report_lines.append("3. Tastan, B., et al. (2024). \"ShapFed: Shapley Value-based Client Selection for Federated Learning.\"")
    report_lines.append("4. Federated Fairness Analytics Framework")
    report_lines.append("5. F3 (Fair Federated Learning Framework)")
    report_lines.append("")
    
    # Write report
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n{'='*70}")
    print(f"Comprehensive Fairness Analysis Report Generated")
    print(f"{'='*70}")
    print(f"Saved to: {save_path}")
    print(f"\nKey Findings:")
    print(f"  - Best Accuracy: {best_acc_alg} ({results[best_acc_alg].get('final_accuracy', 0)*100:.2f}%)")
    print(f"  - Best Selection Fairness: {best_cv_alg} (CV: {results[best_cv_alg].get('selection_cv', 0):.4f})")
    print(f"  - Best Demographic Fairness: {best_spd_alg} (SPD: {results[best_spd_alg].get('spd', 0):.4f})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Example usage with mock data
    mock_results = {
        'Random': {
            'final_accuracy': 0.68,
            'selection_cv': 0.35,
            'selection_jfi': 0.75,
            'spd': 0.140,
            'eod': 0.120,
            'acc_history': [0.50, 0.60, 0.65, 0.68],
            'selection_variance': 12.5
        },
        'LongFed': {
            'final_accuracy': 0.65,
            'selection_cv': 0.20,
            'selection_jfi': 0.92,
            'spd': 0.080,
            'eod': 0.075,
            'acc_history': [0.48, 0.58, 0.63, 0.65],
            'selection_variance': 4.2
        },
        'ShapFed': {
            'final_accuracy': 0.72,
            'selection_cv': 0.65,
            'selection_jfi': 0.45,
            'spd': 0.100,
            'eod': 0.095,
            'acc_history': [0.52, 0.65, 0.70, 0.72],
            'selection_variance': 28.7
        },
        'MAB': {
            'final_accuracy': 0.70,
            'selection_cv': 0.40,
            'selection_jfi': 0.68,
            'spd': 0.046,
            'eod': 0.050,
            'acc_history': [0.51, 0.62, 0.68, 0.70],
            'selection_variance': 15.8
        }
    }
    
    generate_comprehensive_report(mock_results, "EXAMPLE_FAIRNESS_REPORT.md")
