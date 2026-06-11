# Credit Risk Research Lab

Production-style ML pipeline for credit scoring research. Reproducible, modular, and well-tested on Windows.

## Why This Project Exists

Credit scoring is a classic imbalanced classification problem where:
- Default events are rare (typically 5-35% of applications)
- Missing a default (false negative) costs 5x more than rejecting a good client (false positive)
- Accuracy is meaningless - a model predicting "no default" for everyone achieves 95% accuracy but is useless

This project demonstrates a complete ML pipeline from data generation to deployment-ready predictions, with proper attention to threshold optimization, business costs, and explainability.

## Installation (Windows PowerShell)

```powershell
# Clone the repository
git clone <repository-url>
cd credit_risk_lab

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e ".[dev]"

# Optional: install XGBoost/LightGBM
pip install -e ".[boost]"
```

## Project Structure

```
credit_risk_lab/
├── app/
│   ├── __init__.py
│   ├── cli.py              # CLI interface (11 commands)
│   ├── config.py           # Pydantic + YAML configuration
│   ├── constants.py        # Project constants
│   ├── exceptions.py       # Custom exceptions
│   ├── data/
│   │   ├── synthetic_generator.py  # Synthetic data generation
│   │   ├── loader.py              # Data loading
│   │   ├── validation.py          # Data validation
│   │   ├── leakage.py             # Leakage detection
│   │   └── splitting.py           # Data splitting
│   ├── features/
│   │   ├── feature_engineering.py  # 11 engineered features
│   │   └── preprocessing.py        # sklearn Pipeline
│   ├── models/
│   │   ├── baseline.py     # Model definitions
│   │   ├── registry.py     # Model registry + hyperparameter search
│   │   ├── train.py        # Training pipeline
│   │   ├── evaluate.py     # Evaluation metrics
│   │   ├── thresholds.py   # Threshold selection
│   │   ├── calibration.py  # Calibration analysis
│   │   ├── explainability.py  # Permutation importance + reason codes
│   │   └── predict.py      # Prediction module
│   ├── reports/
│   │   ├── eda_report.py   # EDA report
│   │   ├── model_report.py # Model report
│   │   ├── final_report.py # Final comprehensive report
│   │   ├── plots.py        # Plotting utilities
│   │   └── export.py       # HTML export
│   └── utils/
│       ├── io.py           # I/O utilities
│       ├── logging.py      # Logging
│       ├── random.py       # Random state
│       ├── paths.py        # Path utilities
│       └── timing.py       # Timing utilities
├── configs/
│   ├── default.yaml        # Default configuration
│   ├── fast_debug.yaml     # Fast debug mode
│   └── stress.yaml         # Stress test mode
├── tests/
│   ├── unit/               # 33 unit tests
│   ├── integration/        # 5 integration tests
│   ├── smoke/              # 7 smoke tests
│   └── regression/         # 3 regression tests
├── data/
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   └── samples/            # Sample data
├── artifacts/
│   ├── models/             # Trained models
│   ├── reports/            # Generated reports
│   ├── plots/              # Generated plots
│   ├── metrics/            # Evaluation metrics
│   ├── predictions/        # Predictions
│   ├── audit/              # Audit reports
│   └── logs/               # Timing logs
├── pyproject.toml
└── README.md
```

## Full Pipeline Commands

```bash
# Generate synthetic data (5000 rows)
python -m app.cli generate-data --rows 5000 --config configs/default.yaml

# Validate data
python -m app.cli validate-data data/raw/synthetic_credit_data.csv --config configs/default.yaml

# Run EDA
python -m app.cli run-eda data/raw/synthetic_credit_data.csv --config configs/default.yaml

# Train models with hyperparameter search
python -m app.cli train data/raw/synthetic_credit_data.csv --config configs/default.yaml

# Evaluate model
python -m app.cli evaluate --config configs/default.yaml

# Select threshold
python -m app.cli select-threshold --config configs/default.yaml

# Generate explainability report
python -m app.cli explain --config configs/default.yaml

# Make predictions
python -m app.cli predict data/raw/synthetic_credit_data.csv --config configs/default.yaml

# Generate final report
python -m app.cli generate-report --config configs/default.yaml

# Run full pipeline
python -m app.cli run-all --config configs/fast_debug.yaml

# Run technical audit
python -m app.cli audit --config configs/default.yaml
```

## Fast Debug Mode

```bash
# Run complete pipeline with minimal data (500 rows, no hyperparameter search)
python -m app.cli run-all --config configs/fast_debug.yaml
```

## Stress Mode

```bash
# Generate 100k rows and train
python -m app.cli generate-data --rows 100000 --config configs/stress.yaml
python -m app.cli train data/raw/synthetic_credit_data.csv --config configs/stress.yaml
```

## Artifact Descriptions

| Artifact | Description |
|----------|-------------|
| `artifacts/models/best_model.joblib` | Best trained model |
| `artifacts/models/preprocessor.joblib` | Fitted preprocessor |
| `artifacts/models/metadata.json` | Model metadata |
| `artifacts/metrics/metrics.json` | Evaluation metrics |
| `artifacts/metrics/thresholds.json` | Threshold selection results |
| `artifacts/metrics/feature_importance.json` | Feature importance scores |
| `artifacts/reports/final_report.html` | Final portfolio report |
| `artifacts/reports/eda_report.md` | EDA report |
| `artifacts/predictions/predictions.csv` | Predictions with reason codes |

## Metric Explanations

### Why PR-AUC is Primary

PR-AUC (Precision-Recall Area Under Curve) is preferred over ROC-AUC for imbalanced datasets because:
1. It focuses on the minority class (defaults)
2. It doesn't benefit from predicting the majority class
3. It directly measures the trade-off between precision and recall

### Threshold Selection

The pipeline selects thresholds using 4 strategies:
1. **Max F1**: Threshold maximizing F1 score
2. **Target Recall ≥ 0.75**: Ensures 75% of defaults are caught
3. **Target Precision ≥ 0.60**: Ensures 60% of flagged clients actually default
4. **Business Cost Minimization**: Minimizes total expected loss

### Business Cost

- **False Negative (missed default)**: Cost = 5 units
- **False Positive (rejected good client)**: Cost = 1 unit

Missing a default is 5x more costly than rejecting a good client. The business cost threshold minimizes total expected loss.

### Reason Codes

Reason codes explain why a client was flagged as high risk:
- `high_debt_to_income`: DTI > 40%
- `previous_defaults`: Has prior defaults
- `recent_late_payments`: Multiple late payments
- `low_income`: Income < $30k
- `short_credit_history`: < 3 years credit history
- `high_loan_to_income`: Loan > 5x income
- `short_employment`: < 2 years employment
- `high_interest_rate`: Rate > 15%
- `low_savings_buffer`: Savings < $1000
- `high_risk_purpose`: Business or medical loan

## Limitations

1. **Synthetic Data Only**: This project uses synthetic data, not real banking data
2. **No Temporal Features**: No time-series patterns or seasonal effects
3. **No External Data**: No credit bureau, social media, or other external sources
4. **Simplified Business Costs**: Real costs depend on loan amounts, interest rates, and customer lifetime value
5. **No Concept Drift**: No monitoring for model performance degradation

## Future Roadmap

1. Validate with real banking data
2. Add temporal features and survival analysis
3. Implement model monitoring and drift detection
4. Add fairness metrics and bias detection
5. Deploy as REST API for real-time predictions
6. Add A/B testing framework
7. Implement feature store

## Running Tests

```bash
# Run all tests
python -m pytest -q

# Run unit tests only
python -m pytest tests/unit/ -v

# Run integration tests
python -m pytest tests/integration/ -v

# Run smoke tests
python -m pytest tests/smoke/ -v

# Run with coverage
python -m pytest --cov=app --cov-report=term-missing
```

## License

MIT
