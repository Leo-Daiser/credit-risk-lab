# Exploratory Data Analysis Report

## Dataset Overview

- **Rows**: 5000
- **Columns**: 16
- **Target column**: target

## Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| 0 | 2195 | 43.9% |
| 1 | 2805 | 56.1% |

## Missing Values

No missing values found.

## Numeric Features Summary

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| age | 45.23 | 14.14 | 21.00 | 69.00 |
| income | 40920.15 | 21628.07 | 15000.00 | 186875.83 |
| employment_years | 5.03 | 5.06 | 0.00 | 40.00 |
| loan_amount | 30429.36 | 27047.62 | 1000.00 | 200000.00 |
| loan_term_months | 48.03 | 23.77 | 12.00 | 84.00 |
| interest_rate | 14.13 | 6.37 | 3.00 | 25.00 |
| debt_to_income | 0.29 | 0.16 | 0.00 | 0.87 |
| credit_history_years | 8.05 | 7.71 | 0.00 | 40.00 |
| previous_defaults | 0.31 | 0.56 | 0.00 | 4.00 |
| open_credit_lines | 2.99 | 1.72 | 0.00 | 10.00 |
| late_payments_12m | 0.50 | 0.70 | 0.00 | 4.00 |

## Categorical Features Summary

### home_ownership

| Value | Count | Percentage |
|-------|-------|------------|
| mortgage | 2016 | 40.32% |
| rent | 1768 | 35.36% |
| own | 970 | 19.4% |
| other | 246 | 4.92% |

### education

| Value | Count | Percentage |
|-------|-------|------------|
| bachelor | 1806 | 36.12% |
| college | 1225 | 24.5% |
| master | 984 | 19.68% |
| school | 736 | 14.72% |
| phd | 249 | 4.98% |

### purpose

| Value | Count | Percentage |
|-------|-------|------------|
| home | 1296 | 25.92% |
| car | 1014 | 20.28% |
| business | 973 | 19.46% |
| education | 712 | 14.24% |
| other | 503 | 10.06% |
| medical | 502 | 10.04% |

### region

| Value | Count | Percentage |
|-------|-------|------------|
| south | 1034 | 20.68% |
| central | 1030 | 20.6% |
| north | 992 | 19.84% |
| west | 980 | 19.6% |
| east | 964 | 19.28% |

## Target Rate by Categories

### home_ownership

| Value | Default Rate |
|-------|--------------|
| mortgage | 0.5719 |
| other | 0.5650 |
| own | 0.5732 |
| rent | 0.5413 |

### education

| Value | Default Rate |
|-------|--------------|
| bachelor | 0.5576 |
| college | 0.5363 |
| master | 0.5701 |
| phd | 0.5622 |
| school | 0.5978 |

### purpose

| Value | Default Rate |
|-------|--------------|
| business | 0.5694 |
| car | 0.5385 |
| education | 0.5801 |
| home | 0.5741 |
| medical | 0.5737 |
| other | 0.5169 |

### region

| Value | Default Rate |
|-------|--------------|
| central | 0.5738 |
| east | 0.5560 |
| north | 0.5494 |
| south | 0.5735 |
| west | 0.5510 |

## Feature Correlation with Target

| Feature | Correlation |
|---------|-------------|
| debt_to_income | 0.0784 |
| previous_defaults | 0.0665 |
| late_payments_12m | 0.0656 |
| loan_term_months | 0.0267 |
| age | 0.0056 |
| interest_rate | -0.0080 |
| employment_years | -0.0101 |
| open_credit_lines | -0.0151 |
| loan_amount | -0.0152 |
| income | -0.0310 |
| credit_history_years | -0.0481 |
