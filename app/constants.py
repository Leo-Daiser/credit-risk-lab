"""Constants for the Credit Risk Research Lab."""

# Feature columns
REQUIRED_NUMERIC_FEATURES = [
    "age", "income", "employment_years", "loan_amount", "loan_term_months",
    "interest_rate", "debt_to_income", "credit_history_years",
    "previous_defaults", "open_credit_lines", "late_payments_12m",
    "savings_balance", "checking_balance",
]

REQUIRED_CATEGORICAL_FEATURES = [
    "home_ownership", "education", "purpose", "region",
    "application_channel",
]

ALL_RAW_FEATURES = REQUIRED_NUMERIC_FEATURES + REQUIRED_CATEGORICAL_FEATURES + ["applicant_id", "has_cosigner", "target"]

# Target column
TARGET_COLUMN = "target"
ID_COLUMN = "applicant_id"

# Categorical values
HOME_OWNERSHIP_VALUES = ["rent", "mortgage", "own", "other"]
EDUCATION_VALUES = ["school", "college", "bachelor", "master", "phd"]
PURPOSE_VALUES = ["car", "education", "business", "home", "medical", "other"]
REGION_VALUES = ["north", "south", "east", "west", "central"]
CHANNEL_VALUES = ["online", "branch", "partner", "referral"]

# Validation thresholds
MIN_AGE = 18
MAX_AGE = 100
MIN_INCOME = 0
MIN_LOAN_AMOUNT = 0
MIN_EMPLOYMENT_YEARS = 0
MIN_CREDIT_HISTORY_YEARS = 0
MIN_DEBT_TO_INCOME = 0
MAX_DEBT_TO_INCOME = 2.0

# Default rate target for synthetic data
DEFAULT_RATE_TARGET = 0.35

# Business costs
DEFAULT_FN_COST = 5
DEFAULT_FP_COST = 1

# Threshold boundaries
MANUAL_REVIEW_LOW = 0.3
MANUAL_REVIEW_HIGH = 0.7

# Leakage keywords
LEAKAGE_KEYWORDS = ["target", "default", "approved", "decision", "status", "paid", "delinquent", "outcome"]

# Engineered feature names
ENGINEERED_FEATURES = [
    "loan_to_income", "credit_lines_per_history_year", "late_payment_rate",
    "savings_to_loan", "checking_to_loan", "employment_stability_bucket",
    "dti_bucket", "income_bucket", "credit_history_bucket",
    "high_risk_purpose_flag", "has_prior_credit_problem",
]

# Reason code thresholds
REASON_THRESHOLDS = {
    "high_debt_to_income": 0.4,
    "low_income": 30000,
    "recent_late_payments": 2,
    "short_credit_history": 3,
    "high_loan_to_income": 5,
    "short_employment": 2,
    "high_interest_rate": 15.0,
    "low_savings_buffer": 1000,
    "high_risk_purpose": ["business", "medical"],
}
