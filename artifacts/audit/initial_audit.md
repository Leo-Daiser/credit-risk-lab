# Initial Audit: Credit Risk Research Lab

## Current File Tree
```
credit_risk_lab/
├── app/
│   ├── __init__.py
│   ├── cli.py (8 commands, hardcoded paths)
│   ├── config.py (Pydantic + YAML, mostly unused)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── synthetic_generator.py (16 features, no applicant_id)
│   │   └── validation.py (basic checks)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py (4 features)
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── calibration.py (ORPHANED - never called)
│   │   ├── evaluate.py
│   │   ├── explainability.py
│   │   ├── thresholds.py
│   │   └── train.py
│   ├── reports/
│   │   ├── __init__.py
│   │   ├── eda_report.py
│   │   ├── export.py (no code blocks, no bold/italic)
│   │   ├── model_report.py
│   │   └── plots.py (ORPHANED - never called)
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── logging.py
│       └── random.py
├── configs/
│   └── default.yaml
├── tests/
│   ├── __init__.py
│   └── 9 test files (30 tests)
├── data/
│   ├── raw/
│   └── processed/
└── artifacts/
    ├── metrics/
    ├── models/
    └── reports/
```

## Entry Points
- `python -m app.cli <command>` (8 commands)
- `credit-risk` (pyproject.toml script)

## Current CLI Commands
1. generate-data
2. validate-data PATH
3. run-eda PATH
4. train PATH
5. evaluate
6. select-threshold
7. generate-report
8. predict PATH

## Missing from HARD_SPEC
1. **CLI commands**: explain, run-all, audit
2. **Config fields**: applicant_id, id_column, validation_size, default_rate_target, noise_level, missing_rate, outlier_rate, cv_folds, n_jobs, enable_hyperparameter_search, max_search_iter, manual_review_low, manual_review_high, generate_plots, generate_html, generate_markdown, max_missing_share, max_duplicate_share, leakage_keywords
3. **Data columns**: applicant_id, application_channel, has_cosigner, savings_balance, checking_balance
4. **Modules**: constants.py, exceptions.py, data/leakage.py, data/splitting.py, features/schema.py, models/registry.py, models/predict.py, reports/final_report.py, utils/paths.py, utils/timing.py
5. **Configs**: fast_debug.yaml, stress.yaml
6. **Directories**: data/samples/, artifacts/predictions/, artifacts/audit/, artifacts/logs/, tests/unit/, tests/integration/, tests/smoke/, tests/regression/
7. **Tests**: 33+ required tests (currently 30)
8. **Reports**: synthetic_data_card.md, data_validation_report.md, leakage_report.md, threshold_report.md, calibration_report.md, explainability_report.md, stress_test_report.md
9. **Plots**: target_distribution.png, numeric_distributions.png, categorical_target_rates.png, correlation_heatmap.png, missing_values.png, score_distribution.png, threshold_metrics.png, business_cost_curve.png

## Architectural Problems
1. Config-Code Disconnect: CONFIG.paths.* never used by code
2. 16+ hardcoded relative paths in cli.py
3. Orphaned modules: calibration.py, plots.py functions
4. Unused code: reason_mapping dict, get_engineered_feature_names()
5. CLI select_threshold ignores --strategy option
6. No cross-validation
7. No hyperparameter search
8. No timing/metrics tracking

## Leakage Risks
1. No dedicated leakage detection module
2. No applicant_id handling
3. No duplicate detection across train/test

## Reproducibility Problems
1. No save/load of test indices
2. No versioning of artifacts
3. No timing reports

## Windows Compatibility
- No issues detected (paths use Path objects or forward slashes)

## Dependency Problems
- No issues detected

## Prioritized Improvement Plan
1. P0: Fix config-code disconnect, use CONFIG.paths
2. P0: Add missing CLI commands (explain, run-all, audit)
3. P0: Add leakage detection module
4. P0: Add model registry with hyperparameter search
5. P1: Update synthetic generator with required columns
6. P1: Add fast_debug.yaml and stress.yaml configs
7. P1: Reorganize tests into unit/integration/smoke/regression
8. P1: Generate all required artifacts and reports
9. P2: Add timing utilities
10. P2: Add constants.py and exceptions.py
11. P2: Generate all required plots
