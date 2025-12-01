"""
Configuration file for Credit Risk Modeling System
Contains all paths, parameters, and business rules
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Model paths
BEST_MODEL_PATH = PROCESSED_DATA_DIR / "best_pd_model.h5"
SCALER_PATH = PROCESSED_DATA_DIR / "scaler.pkl"
MODEL_COMPARISON_PATH = PROCESSED_DATA_DIR / "all_models_comparison.csv"

# Data file paths
LOANS_CLEANED_PATH = PROCESSED_DATA_DIR / "loans_cleaned_final.csv"
EXPECTED_LOSS_PATH = PROCESSED_DATA_DIR / "expected_loss_results.csv"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "test_predictions.csv"

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Best model information
BEST_MODEL_NAME = "Neural Network"
BEST_MODEL_AUC = 0.7939

# Model comparison results
MODEL_PERFORMANCE = {
    "Logistic Regression": {"AUC-ROC": 0.7877, "Accuracy": 0.8368},
    "Random Forest": {"AUC-ROC": 0.7871, "Accuracy": 0.8364},
    "XGBoost": {"AUC-ROC": 0.7902, "Accuracy": 0.8369},
    "LightGBM": {"AUC-ROC": 0.7896, "Accuracy": 0.8356},
    "Gradient Boosting": {"AUC-ROC": 0.7909, "Accuracy": 0.8374},
    "Neural Network": {"AUC-ROC": 0.7939, "Accuracy": 0.8370},
    "Stacking Ensemble": {"AUC-ROC": 0.7929, "Accuracy": 0.8355}
}

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/test/validation split ratios
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ============================================================================
# BUSINESS RULES & THRESHOLDS
# ============================================================================

# Credit decision thresholds (Probability of Default)
CREDIT_THRESHOLDS = {
    "approve": 0.3,  # PD < 30% → Approve
    "review": 0.5,  # 30% ≤ PD < 50% → Manual Review
    "reject": 0.5  # PD ≥ 50% → Reject
}

# Credit score mapping (300-850 range)
CREDIT_SCORE_RANGES = {
    "Poor": (300, 579),
    "Fair": (580, 669),
    "Good": (670, 739),
    "Very Good": (740, 799),
    "Excellent": (800, 850)
}

# Interest rate adjustments based on risk
INTEREST_RATE_BASE = 0.10  # 10% base rate
INTEREST_RATE_MULTIPLIERS = {
    "Excellent": 0.8,  # 8% (10% * 0.8)
    "Very Good": 0.9,  # 9%
    "Good": 1.0,  # 10%
    "Fair": 1.2,  # 12%
    "Poor": 1.5  # 15%
}

# Loan amount limits based on risk category
LOAN_AMOUNT_LIMITS = {
    "Excellent": 50000,
    "Very Good": 40000,
    "Good": 30000,
    "Fair": 20000,
    "Poor": 10000
}

# ============================================================================
# BUSINESS IMPACT METRICS
# ============================================================================

# Time savings
MANUAL_REVIEW_TIME_MINUTES = 30
AI_DECISION_TIME_SECONDS = 3

# Cost savings
ANALYST_HOURLY_RATE = 35  # USD per hour
ANNUAL_LOAN_APPLICATIONS = 50000

# Revenue impact
AVERAGE_LOAN_AMOUNT = 15000
DEFAULT_COST_MULTIPLIER = 1.5  # Cost when loan defaults

# ============================================================================
# FAIRNESS & COMPLIANCE
# ============================================================================

# Protected attributes (for fairness testing)
PROTECTED_ATTRIBUTES = [
    "person_gender",
    "person_age_group",
    "person_home_ownership"
]

# Fairness thresholds
DEMOGRAPHIC_PARITY_THRESHOLD = 0.10  # 10% max difference
EQUAL_OPPORTUNITY_THRESHOLD = 0.10  # 10% max difference in TPR

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Chart colors
COLOR_PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#d62728",
    "info": "#9467bd"
}

# Plot style
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_SIZE = (12, 6)
DPI = 300

# ============================================================================
# STREAMLIT APP SETTINGS
# ============================================================================

APP_TITLE = "Credit Risk Assessment System"
APP_ICON = "💳"
PAGE_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ============================================================================
# FEATURE NAMES (for model input)
# ============================================================================

# These should match your processed data columns
REQUIRED_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    # Add all other features your model expects
]

# Feature display names (for UI)
FEATURE_DISPLAY_NAMES = {
    "person_age": "Age",
    "person_income": "Annual Income",
    "person_emp_length": "Employment Length (years)",
    "loan_amnt": "Loan Amount",
    "loan_int_rate": "Interest Rate",
    "loan_percent_income": "Loan as % of Income",
    "cb_person_cred_hist_length": "Credit History Length (years)",
    "credit_score": "Credit Score"
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_credit_decision(probability_of_default: float) -> str:
    """
    Determine credit decision based on PD threshold

    Args:
        probability_of_default: Probability of default (0-1)

    Returns:
        Decision: "Approve", "Manual Review", or "Reject"
    """
    if probability_of_default < CREDIT_THRESHOLDS["approve"]:
        return "Approve"
    elif probability_of_default < CREDIT_THRESHOLDS["review"]:
        return "Manual Review"
    else:
        return "Reject"


def calculate_credit_score(probability_of_default: float) -> int:
    """
    Convert PD to credit score (300-850 range)

    Args:
        probability_of_default: Probability of default (0-1)

    Returns:
        Credit score between 300 and 850
    """
    # Inverse relationship: lower PD = higher score
    score = 850 - (probability_of_default * 550)
    return int(max(300, min(850, score)))


def get_risk_category(credit_score: int) -> str:
    """
    Get risk category from credit score

    Args:
        credit_score: Credit score (300-850)

    Returns:
        Risk category: "Poor", "Fair", "Good", "Very Good", or "Excellent"
    """
    for category, (min_score, max_score) in CREDIT_SCORE_RANGES.items():
        if min_score <= credit_score <= max_score:
            return category
    return "Unknown"


def calculate_interest_rate(risk_category: str) -> float:
    """
    Calculate interest rate based on risk category

    Args:
        risk_category: Risk category from get_risk_category()

    Returns:
        Interest rate (decimal)
    """
    multiplier = INTEREST_RATE_MULTIPLIERS.get(risk_category, 1.0)
    return INTEREST_RATE_BASE * multiplier


def get_max_loan_amount(risk_category: str) -> int:
    """
    Get maximum loan amount for risk category

    Args:
        risk_category: Risk category from get_risk_category()

    Returns:
        Maximum loan amount in USD
    """
    return LOAN_AMOUNT_LIMITS.get(risk_category, 10000)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_paths():
    """Validate that all required paths exist"""
    required_paths = [
        BEST_MODEL_PATH,
        SCALER_PATH,
        LOANS_CLEANED_PATH
    ]

    missing_paths = []
    for path in required_paths:
        if not path.exists():
            missing_paths.append(str(path))

    if missing_paths:
        raise FileNotFoundError(
            f"Missing required files:\n" + "\n".join(missing_paths)
        )

    return True


# ============================================================================
# MEDICAL LOAN CONFIGURATIONS
# ============================================================================

MEDICAL_LOAN_CONFIG = {
    # Basic info
    'loan_type': 'medical_procedure',
    'name': 'Medical Procedure Financing',
    'name_es': 'Financiamiento de Procedimientos Médicos',
    'description': 'Financiamiento para procedimientos médicos no cubiertos por seguros',

    # Clinic commission (from research)
    'clinic_commission_rate': 0.05,  # 5% - Industry standard
    'commission_range': (0.03, 0.06),  # 3-6% industry range
    'commission_source': 'Industry research: standard clinic commission 5%, range 3-6%',

    # Interest rate component adjustments (vs standard personal loans)
    'component_adjustments': {
        'credit_risk_reduction': -0.005,  # -0.5% (verified use, direct payment to provider)
        'liquidity_reduction': -0.003,  # -0.3% (predictable cash flows, fixed costs)
        'operating_reduction': -0.005,  # -0.5% (clinic partnerships eliminate marketing)
        'profit_adjustment': 0.000,  # 0% (same profit margin)
        'total_reduction': -0.013  # ~1.3% total savings
    },

    # Justifications for lower risk (for presentation/documentation)
    'lower_risk_justifications': [
        'Pago directo a proveedor médico - elimina riesgo de fraude',
        'Uso verificado de fondos - no puede usarse para otros fines',
        'Priorización de gastos médicos en presupuesto familiar',
        'Validación del proveedor - confirma realización del procedimiento',
        'Costos fijos y predecibles - reduce incertidumbre de flujo de caja'
    ],

    # Loan terms
    'min_amount_mxn': 5000,
    'max_amount_mxn': 300000,
    'min_term_months': 6,
    'max_term_months': 24,  # vs 18 for standard (longer terms available)
    'grace_period_months': 1,  # Optional 1-month grace period
    'payment_method': 'direct_to_provider',  # Always paid to clinic, never to borrower

    # Approval thresholds (slightly more lenient due to lower risk profile)
    'thresholds': {
        'approve': 0.35,  # vs 0.30 for standard (can approve slightly higher PD)
        'review': 0.50,  # Same as standard
        'reject': 0.50  # Reject if PD >= 50%
    },

    # Mexican healthcare market context (from research)
    'market_data': {
        'mexico_oop_percentage': 0.41,  # 41% vs 18-23% OECD average
        'oecd_average_oop': 0.20,  # ~20% for comparison
        'households_with_medical_expenses': 0.677,  # 67.7% (ENIGH 2020)
        'catastrophic_expense_rate': 0.04,  # 4% of households
        'aesthetic_market_size_usd_2025': 1_457_800_000,  # $1.46B
        'aesthetic_market_size_usd_2034': 3_100_000_000,  # $3.1B projected
        'annual_growth_rate': 0.09,  # 9%
        'procedures_per_year': 848_000,  # 2022 data
        'medical_inflation_rate': 0.149,  # 14.9% vs general inflation
        'source': 'OECD Health Statistics, ENIGH 2020, Industry Research'
    },

    # Example procedures (for documentation/presentation, not for rate differentiation)
    'common_procedures': [
        'Procedimientos dentales (implantes, ortodoncia)',
        'Cirugía estética (liposucción, rinoplastia)',
        'Tratamientos dermatológicos especializados',
        'Cirugía ocular (LASIK, corrección de visión)',
        'Tratamientos de fertilidad (FIV)',
        'Cirugía bariátrica'
    ]
}

# Business thresholds for medical loans
MEDICAL_BUSINESS_THRESHOLDS = {
    'min_monthly_payment': 500,  # MXN
    'max_dti_ratio': 0.45,  # 45% debt-to-income (vs 40% standard, more lenient)
    'min_procedure_amount': 5000,  # MXN
    'max_procedure_amount': 300000,  # MXN
}


if __name__ == "__main__":
    # Test configuration
    print("Credit Risk System Configuration")
    print("=" * 50)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Best Model: {BEST_MODEL_NAME} (AUC: {BEST_MODEL_AUC})")
    print(f"Model Path: {BEST_MODEL_PATH}")
    print("\nValidating paths...")
    try:
        validate_paths()
        print("All required files found")
    except FileNotFoundError as e:
        print(f"{e}")