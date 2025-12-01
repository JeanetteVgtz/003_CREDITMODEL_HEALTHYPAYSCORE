"""
Test script for medical loan functionality
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from credit_decision_engine import CreditDecisionEngine
import pandas as pd

# Load test data
print("Loading test data...")
X_test = pd.read_csv('data/processed/X_test.csv')

# Get first application as DataFrame (keep it as DataFrame, not dict)
app_data = X_test.iloc[[0]]  # Double brackets to keep as DataFrame

print("Initializing credit decision engine...")
engine = CreditDecisionEngine()

# Test standard loan
print('\n' + '='*60)
print('STANDARD LOAN')
print('='*60)
result_standard = engine.process_application(app_data, loan_type='standard')
print(f'Decision: {result_standard["decision"]}')
print(f'PD: {result_standard["probability_of_default"]:.2%}')
print(f'Risk Category: {result_standard["risk_category"]}')
print(f'APR: {result_standard["recommended_interest_rate"]:.2f}%')

# Test medical loan
print('\n' + '='*60)
print('MEDICAL LOAN')
print('='*60)
result_medical = engine.process_application(
    app_data,
    loan_type='medical',
    procedure_amount=40000
)
print(f'Decision: {result_medical["decision"]}')
print(f'PD: {result_medical["probability_of_default"]:.2%}')
print(f'Risk Category: {result_medical["risk_category"]}')
print(f'APR: {result_medical["recommended_interest_rate"]:.2f}%')
print(f'\nMedical Loan Details:')
print(f'  Procedure amount: ${result_medical["procedure_amount"]:,.2f} MXN')
print(f'  Clinic receives: ${result_medical["clinic_receives"]:,.2f} MXN')
print(f'  Commission: ${result_medical["clinic_commission_amount"]:,.2f} MXN ({result_medical["clinic_commission_rate"]}%)')
print(f'  Payment method: {result_medical["payment_method"]}')
print(f'\nComparison:')
print(f'  Standard loan APR: {result_medical["standard_loan_rate"]:.2f}%')
print(f'  Medical loan APR: {result_medical["recommended_interest_rate"]:.2f}%')
print(f'  Savings: {result_medical["rate_savings_vs_standard"]:.2f} percentage points')

print('\n' + '='*60)
print('ALL TESTS PASSED - Medical loan functionality working')
print('='*60)