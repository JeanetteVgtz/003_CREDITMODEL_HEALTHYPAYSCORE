"""
Interest Rate Calculator - Based on Actual Market Data (October 2025)
"""


def calculate_interest_rate(probability_of_default: float) -> float:
    """
    Calculate risk-based interest rate using actual market structure.

    Formula: APR = Base Components + Credit Risk Premium

    Base Components (Fixed):
    - Federal Reserve Rate: 3.875% (midpoint of 3.75-4.00%)
    - Inflation Premium: 3.0%
    - Liquidity Premium: 1.0%
    - Operating Costs: 2.0%
    - Profit Margin: 3.0%
    TOTAL BASE: 12.875%

    Credit Risk Premium (Variable, based on PD):
    - PD < 10%: +1.0% (Excellent risk)
    - PD 10-20%: +3.5% (Good risk)
    - PD 20-35%: +7.5% (Fair risk)
    - PD > 35%: +14.5% (Poor risk)

    Examples:
    - PD = 5% → APR = 12.875% + 1.0% = 13.88%
    - PD = 15% → APR = 12.875% + 3.5% = 16.38%
    - PD = 25% → APR = 12.875% + 7.5% = 20.38%
    - PD = 40% → APR = 12.875% + 14.5% = 27.38%

    Args:
        probability_of_default: Float between 0 and 1

    Returns:
        Annual interest rate as decimal (e.g., 0.1388 for 13.88%)
    """
    # Base components (fixed for all loans)
    BASE_RATE = 0.03875  # Federal Reserve midpoint
    INFLATION = 0.03  # US inflation rate (Oct 2025)
    LIQUIDITY = 0.01  # Liquidity premium
    OPERATING = 0.02  # Operating costs
    PROFIT = 0.03  # Profit margin

    base_total = BASE_RATE + INFLATION + LIQUIDITY + OPERATING + PROFIT  # 12.875%

    # Credit risk premium (variable based on PD)
    if probability_of_default < 0.10:  # Excellent: PD < 10%
        credit_risk_premium = 0.01  # +1.0%
    elif probability_of_default < 0.20:  # Good: PD 10-20%
        credit_risk_premium = 0.035  # +3.5%
    elif probability_of_default < 0.35:  # Fair: PD 20-35%
        credit_risk_premium = 0.075  # +7.5%
    else:  # Poor: PD >= 35%
        credit_risk_premium = 0.145  # +14.5%

    total_apr = base_total + credit_risk_premium

    # Cap at 36% (regulatory limit in most states)
    total_apr = min(total_apr, 0.36)

    return round(total_apr, 4)  # Round to 4 decimals for precision


def get_risk_tier(probability_of_default: float) -> str:
    """
    Get risk tier category based on PD.

    Args:
        probability_of_default: Float between 0 and 1

    Returns:
        Risk tier label
    """
    if probability_of_default < 0.10:
        return "Excellent"
    elif probability_of_default < 0.20:
        return "Good"
    elif probability_of_default < 0.35:
        return "Fair"
    else:
        return "Poor"


def calculate_medical_loan_rate(pd: float) -> dict:
    """
    Calculate interest rate for medical procedure loans.

    Medical loans receive lower rates due to:
    - Direct payment to medical provider (verified use)
    - Health expense prioritization in household budgets
    - Clinic partnerships reducing operating costs
    - Predictable cash flows (fixed procedure costs)

    Args:
        pd: Probability of default (0-1)

    Returns:
        dict with:
            - total_apr: Total annual percentage rate (decimal)
            - components: Breakdown of rate components
            - savings_vs_standard: Difference from standard loan rate
    """
    from config import MEDICAL_LOAN_CONFIG

    # Base components (same as standard loans)
    BASE_RATE = 0.03875  # Federal Reserve midpoint
    INFLATION = 0.03  # Current inflation expectation
    PROFIT = 0.03  # Profit margin (same as standard)

    # Calculate credit risk premium based on PD
    if pd < 0.10:
        credit_risk_premium = 0.01  # 1% for excellent credit
    elif pd < 0.20:
        credit_risk_premium = 0.035  # 3.5% for good credit
    elif pd < 0.35:
        credit_risk_premium = 0.075  # 7.5% for fair credit
    else:
        credit_risk_premium = 0.145  # 14.5% for poor credit

    # Apply medical loan adjustments
    adjustments = MEDICAL_LOAN_CONFIG['component_adjustments']

    # Adjusted components
    credit_risk_adjusted = credit_risk_premium + adjustments['credit_risk_reduction']
    liquidity_adjusted = 0.01 + adjustments['liquidity_reduction']  # Standard 1% - 0.3%
    operating_adjusted = 0.02 + adjustments['operating_reduction']  # Standard 2% - 0.5%

    # Calculate total APR
    total_apr = (
            BASE_RATE +
            INFLATION +
            credit_risk_adjusted +
            liquidity_adjusted +
            operating_adjusted +
            PROFIT
    )

    # Calculate standard loan rate for comparison
    standard_apr = calculate_interest_rate(pd)

    # Component breakdown
    components = {
        'base_rate': BASE_RATE,
        'inflation': INFLATION,
        'credit_risk': credit_risk_adjusted,
        'liquidity': liquidity_adjusted,
        'operating': operating_adjusted,
        'profit': PROFIT,
        'total': total_apr
    }

    return {
        'total_apr': total_apr,
        'total_apr_percent': round(total_apr * 100, 2),
        'components': components,
        'standard_apr': standard_apr,
        'savings_vs_standard': standard_apr - total_apr,
        'savings_percent': round(((standard_apr - total_apr) / standard_apr) * 100, 2)
    }


def compare_loan_types(pd: float) -> dict:
    """
    Compare standard personal loan vs medical loan for same applicant.

    Args:
        pd: Probability of default (0-1)

    Returns:
        dict with comparison metrics
    """
    standard_apr = calculate_interest_rate(pd)
    medical_result = calculate_medical_loan_rate(pd)

    return {
        'probability_of_default': pd,
        'standard_loan': {
            'apr': standard_apr,
            'apr_percent': round(standard_apr * 100, 2)
        },
        'medical_loan': {
            'apr': medical_result['total_apr'],
            'apr_percent': medical_result['total_apr_percent']
        },
        'savings': {
            'absolute': medical_result['savings_vs_standard'],
            'absolute_percent': round(medical_result['savings_vs_standard'] * 100, 2),
            'relative_percent': medical_result['savings_percent']
        }
    }


def calculate_clinic_commission(procedure_amount: float) -> dict:
    """
    Calculate clinic commission for medical loan.

    Args:
        procedure_amount: Cost of medical procedure in MXN

    Returns:
        dict with commission details
    """
    from config import MEDICAL_LOAN_CONFIG

    commission_rate = MEDICAL_LOAN_CONFIG['clinic_commission_rate']
    commission = procedure_amount * commission_rate
    clinic_receives = procedure_amount - commission

    return {
        'procedure_amount': procedure_amount,
        'commission_rate': commission_rate,
        'commission_rate_percent': round(commission_rate * 100, 1),
        'commission_amount': round(commission, 2),
        'clinic_receives': round(clinic_receives, 2),
        'lender_retains': round(commission, 2)
    }


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("MEDICAL LOAN RATE CALCULATOR - TEST")
    print("=" * 60)

    test_pds = [0.05, 0.15, 0.25, 0.40]
    pd_labels = ["Excellent (5%)", "Good (15%)", "Fair (25%)", "Poor (40%)"]

    for pd, label in zip(test_pds, pd_labels):
        print(f"\n{label} Credit:")
        print("-" * 60)

        comparison = compare_loan_types(pd)

        print(f"Standard Loan APR: {comparison['standard_loan']['apr_percent']:.2f}%")
        print(f"Medical Loan APR:  {comparison['medical_loan']['apr_percent']:.2f}%")
        print(
            f"Savings: {comparison['savings']['absolute_percent']:.2f} percentage points ({comparison['savings']['relative_percent']:.1f}% reduction)")

    print("\n" + "=" * 60)
    print("CLINIC COMMISSION EXAMPLE")
    print("=" * 60)

    commission_info = calculate_clinic_commission(40000)
    print(f"\nProcedure amount: ${commission_info['procedure_amount']:,.2f} MXN")
    print(f"Commission rate: {commission_info['commission_rate_percent']}%")
    print(f"Commission amount: ${commission_info['commission_amount']:,.2f} MXN")
    print(f"Clinic receives: ${commission_info['clinic_receives']:,.2f} MXN")
    print(f"Lender retains: ${commission_info['lender_retains']:,.2f} MXN")

# Test the function
if __name__ == "__main__":
    print("=" * 80)
    print("INTEREST RATE CALCULATOR TEST")
    print("=" * 80)

    test_pds = [0.05, 0.15, 0.25, 0.40, 0.50]

    for pd in test_pds:
        apr = calculate_interest_rate(pd)
        tier = get_risk_tier(pd)
        print(f"PD: {pd * 100:5.1f}% | Tier: {tier:10s} | APR: {apr * 100:6.2f}%")



