# FEMNIST Fair Client Selection - Comprehensive Fairness Metrics

## Dataset: FEMNIST (Federated Extended MNIST)

**NOT FashionMNIST!** FEMNIST is specifically designed for federated learning:

- **62 classes**: 10 digits (0-9) + 26 uppercase letters (A-Z) + 26 lowercase letters (a-z)
- **3,597 writers**: Naturally partitioned by writer_id (each writer's unique handwriting style)
- **Non-IID distribution**: Different writers prefer different characters, creating realistic heterogeneity
- **Grayscale 28×28 images**: Handwritten characters

## Fairness Metrics Implemented

### 1. Statistical Parity Difference (SPD)
**Formula**: `SPD = |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|`

**Purpose**: Demographic fairness - ensures equal positive outcome rates across protected groups

**Interpretation**:
- SPD ≈ 0: Perfect fairness
- SPD > 0.1: Significant bias
- Target: < 0.05 for fair systems

**Literature Benchmark** (Bouzamoucha et al., 2025):
- Random baseline: SPD = 0.140
- MAB algorithm: SPD = 0.046 (67.1% reduction)

### 2. Equalised Odds
**Formula**: Equal TPR and FPR across groups

**Purpose**: Most robust group fairness metric

**Components**:
- TPR Difference: `|TPR(A=0) - TPR(A=1)|`
- FPR Difference: `|FPR(A=0) - FPR(A=1)|`
- Combined Score: `(TPR_diff + FPR_diff) / 2`

**Interpretation**:
- 0: Perfect fairness
- > 0.1: Significant disparity in error rates

### 3. Equal Opportunity Difference (EOD)
**Formula**: `|P(Ŷ=1|Y=1,A=0) - P(Ŷ=1|Y=1,A=1)|`

**Purpose**: Simplified fairness focusing only on true positive rates

**Interpretation**:
- 0: Equal opportunity
- > 0.1: One group has better outcomes when qualified

### 4. Jain's Fairness Index (JFI)
**Formula**: `JFI = (Σx_i)² / (n × Σx_i²)`

**Purpose**: Performance uniformity across clients

**Range**: (0, 1] where 1 = perfect uniformity

**Interpretation**:
- JFI ≥ 0.8: Generally fair
- JFI < 0.5: Significant inequality
- JFI → 0: Some clients dominate

### 5. Coefficient of Variation (CV)
**Formula**: `CV = std / mean`

**Purpose**: Normalized disparity measure for selection fairness

**Interpretation**:
- CV = 0: Perfect uniformity
- CV < 0.5: Fair distribution
- CV > 1.0: High inequality

### 6. Four-Notion Framework

#### a) Individual Fairness
- **Metric**: JFI of gain ratios (performance / contribution)
- **Purpose**: Clients perform proportionally to their contribution

#### b) Incentive Fairness
- **Metric**: JFI of reward-to-contribution ratios
- **Purpose**: Clients rewarded proportionally to contributions

#### c) Orchestrator Fairness
- **Metric**: Mean normalized client performance
- **Purpose**: Server maximizes objective function

#### d) Protected Group Fairness
- **Metric**: Aggregated equalised odds across sensitive attributes
- **Purpose**: Equitable performance for protected groups

## Algorithms Tested

### 1. Random Selection (Baseline)
- **Method**: Uniform random sampling
- **Expected**: Moderate fairness by chance
- **Benchmark SPD**: 0.14-0.20
- **Use Case**: Simple, unbiased baseline

### 2. LongFed (Li et al., 2024)
- **Method**: Lyapunov optimization with virtual queues
- **Objective**: Individual fairness (balanced participation)
- **Expected**: Lowest selection CV (best JFI)
- **Guarantee**: (1-1/e) approximation for submodular maximization
- **Trade-off**: May sacrifice accuracy for fairness

### 3. ShapFed (Tastan et al., 2024)
- **Method**: Shapley value approximation (cosine similarity)
- **Objective**: Maximize global accuracy
- **Expected**: Highest accuracy, higher CV
- **Strength**: Identifies high-quality contributors
- **Trade-off**: Can lead to client starvation

### 4. MAB (Bouzamoucha et al., 2025)
- **Method**: Multi-armed bandit with epsilon-greedy
- **Objective**: Demographic fairness
- **Expected**: Lowest SPD/EOD
- **Benchmark**: 67.1% bias reduction
- **Trade-off**: Needs sensitive attributes

## Expected Trade-offs

### Accuracy vs Fairness Spectrum

```
High Accuracy,     Balanced         High Fairness,
Low Fairness                        Lower Accuracy
    |                |                    |
 ShapFed          MAB/Random          LongFed
```

### Metric Trade-offs

| Algorithm | Accuracy | Selection Fairness | Demographic Fairness |
|-----------|----------|-------------------|---------------------|
| Random    | Medium   | Medium            | Medium              |
| LongFed   | Lower    | **Highest** (JFI) | Medium              |
| ShapFed   | **Highest** | Lowest         | Medium              |
| MAB       | High     | Medium            | **Highest** (SPD/EOD) |

## Validation Against Literature

### Expected Results

1. **MAB SPD Reduction**: 50-70% vs Random
2. **LongFed Selection CV**: Lowest among all algorithms
3. **ShapFed Accuracy**: Highest among all algorithms
4. **Random SPD**: 0.14-0.20 range

### Success Criteria

- ✓ MAB achieves SPD < 0.10
- ✓ LongFed has lowest selection CV
- ✓ ShapFed has highest final accuracy
- ✓ All algorithms show improvement over synthetic random data

## Experimental Setup

- **Clients**: 10-20 writers
- **Rounds**: 15-30 federated rounds
- **Clients per Round**: 3-5 selected
- **Local Epochs**: 2
- **Model**: SimpleCNN (1 channel, 28×28 input, 62 classes)
- **Optimizer**: SGD with lr=0.01

## Output Files

1. **femnist_results.png**: Visualization comparing all algorithms
2. **femnist_results.json**: Raw metrics for further analysis
3. **FAIRNESS_ANALYSIS_REPORT.md**: Comprehensive trade-off analysis

## References

1. Bouzamoucha, M., et al. (2025). "Multi-Armed Bandit Approach for Fair Client Selection in Federated Learning."
2. Li, T., et al. (2024). "LongFed: Longitudinal Fairness in Federated Learning via Lyapunov Optimization."
3. Tastan, B., et al. (2024). "ShapFed: Shapley Value-based Client Selection for Federated Learning."
4. Federated Fairness Analytics Framework
5. F3 (Fair Federated Learning Framework)
