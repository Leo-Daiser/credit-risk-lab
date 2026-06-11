# Model Training Report

## Model Comparison

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Brier Score |
|-------|---------|--------|-----|-----------|--------|-------------|
| logistic_regression | 0.5724 | 0.6233 | 0.5598 | 0.6154 | 0.5134 | 0.2466 |
| random_forest | 0.5361 | 0.6012 | 0.6472 | 0.5680 | 0.7522 | 0.2485 |
| gradient_boosting | 0.5426 | 0.5996 | 0.6377 | 0.5722 | 0.7201 | 0.2513 |

## Selected Model

**unknown** selected based on PR-AUC (primary metric for imbalanced datasets).

## Metrics Interpretation

- **ROC-AUC (0.5724)**: Measures ability to distinguish between classes. 1.0 = perfect, 0.5 = random.
- **PR-AUC (0.6233)**: Area under Precision-Recall curve. More informative than ROC-AUC for imbalanced datasets.
- **F1 (0.5598)**: Harmonic mean of precision and recall. Balances false positives and false negatives.
- **Precision (0.6154)**: Proportion of predicted defaults that were actual defaults.
- **Recall (0.5134)**: Proportion of actual defaults that were correctly identified.
- **Brier Score (0.2466)**: Measures prediction accuracy. Lower is better.

## Why Accuracy is Not the Main Metric

In credit risk modeling, the target variable is typically imbalanced (most clients don't default). With 90% non-defaults, a model that predicts 'no default' for everyone achieves 90% accuracy but is completely useless.

PR-AUC is preferred because:
- It focuses on the minority class (defaults)
- It doesn't benefit from predicting the majority class
- It directly measures the trade-off between precision and recall

## Threshold Selection

| Strategy | Threshold |
|----------|-----------|
| max_f1 | 0.330 |
| target_recall_075 | 0.450 |
| target_precision_060 | 0.480 |
| business_cost | 0.290 |

**Recommended threshold**: 0.290 (business cost minimization)

### Business Cost Rationale

- False negative (missed default): cost = 5 units
- False positive (rejected good client): cost = 1 unit
- Missing a default is 5x more costly than rejecting a good client
- Business cost threshold minimizes total expected loss
