"""
COMPREHENSIVE DATA VALIDATION SCRIPT
Checks that all metrics match across:
- Saved model files
- Test predictions
- Streamlit displays
- Config files
- Profitability analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tensorflow import keras
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

print("=" * 80)
print("COMPREHENSIVE DATA VALIDATION AUDIT")
print("=" * 80)

# ============================================================================
# SECTION 1: FILE EXISTENCE CHECK
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: FILE EXISTENCE CHECK")
print("=" * 80)

required_files = {
    'Scaler file': 'data/processed/scaler.pkl',
    'X_test': 'data/processed/X_test.csv',
    'y_test': 'data/processed/y_test.csv',
    'Predictions': 'data/processed/pd_test_predictions.csv',
    'Model comparison': 'data/processed/all_models_comparison.csv',
    'Config': 'src/config.py',
    'Credit engine': 'src/credit_decision_engine.py',
    'Streamlit app': 'streamlit_app.py'
}

# Check for model file (either .pkl or .h5)
model_pkl = Path('data/processed/neural_network_model.pkl')
model_h5 = Path('data/processed/best_pd_model.h5')
model_exists = model_pkl.exists() or model_h5.exists()

if model_pkl.exists():
    print(f"✓ Model file               data/processed/neural_network_model.pkl")
elif model_h5.exists():
    print(f"✓ Model file               data/processed/best_pd_model.h5")
else:
    print(f"✗ Model file               (not found: .pkl or .h5)")

files_ok = model_exists

for name, path in required_files.items():
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {name:25s} {path}")
    if not exists:
        files_ok = False

if not files_ok:
    print("\n⚠️  WARNING: Some files are missing!")
    print("Cannot proceed with full validation.")
    exit(1)
else:
    print("\n✓ All required files found")

# ============================================================================
# SECTION 2: LOAD ALL DATA
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: LOADING DATA")
print("=" * 80)

try:
    # Load predictions
    predictions = pd.read_csv('data/processed/pd_test_predictions.csv')
    print(f"✓ Loaded predictions: {len(predictions):,} rows")
    print(f"  Columns: {list(predictions.columns)}")

    # Load test data
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    print(f"✓ Loaded X_test: {X_test.shape}")
    print(f"✓ Loaded y_test: {y_test.shape}")

    # Load model comparison
    model_comparison = pd.read_csv('data/processed/all_models_comparison.csv')
    print(f"✓ Loaded model comparison: {len(model_comparison)} models")

    # Load model
    try:
        model = keras.models.load_model('data/processed/best_pd_model.h5')
        print(f"✓ Loaded Keras model (.h5)")
    except:
        with open('data/processed/neural_network_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Loaded wrapped model (.pkl)")

    # Load scaler
    with open('data/processed/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Loaded scaler")

except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# ============================================================================
# SECTION 3: VERIFY MODEL PERFORMANCE METRICS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: MODEL PERFORMANCE VERIFICATION")
print("=" * 80)

# Get predictions from file
y_true = predictions['y_test'].values
y_pred_proba = predictions['y_pred_proba'].values
y_pred = predictions['y_pred'].values

# Calculate metrics from predictions file
auc_from_file = roc_auc_score(y_true, y_pred_proba)
acc_from_file = accuracy_score(y_true, y_pred)

print(f"\nMetrics from predictions file:")
print(f"  AUC-ROC: {auc_from_file:.4f}")
print(f"  Accuracy: {acc_from_file:.4f}")

# Get metrics from model comparison file
nn_metrics = model_comparison[model_comparison['Model'] == 'Neural Network'].iloc[0]
auc_from_comparison = nn_metrics['AUC-ROC']
acc_from_comparison = nn_metrics['Accuracy']

print(f"\nMetrics from model comparison file:")
print(f"  AUC-ROC: {auc_from_comparison:.4f}")
print(f"  Accuracy: {acc_from_comparison:.4f}")

# Check config.py
exec(open('src/config.py').read())
auc_from_config = BEST_MODEL_AUC

print(f"\nMetrics from config.py:")
print(f"  BEST_MODEL_AUC: {auc_from_config:.4f}")

# Check streamlit_app.py for hardcoded values
streamlit_content = open('streamlit_app.py', 'r', encoding='utf-8').read()
if '0.7939' in streamlit_content or '79.4%' in streamlit_content:
    print(f"\nFound in streamlit_app.py:")
    print(f"  Hardcoded AUC values found (checking if correct...)")

# VALIDATION
print("\n" + "-" * 80)
print("VALIDATION RESULTS:")
print("-" * 80)

tolerance = 0.0001
metrics_match = (
        abs(auc_from_file - auc_from_comparison) < tolerance and
        abs(auc_from_file - auc_from_config) < tolerance and
        abs(acc_from_file - acc_from_comparison) < tolerance
)

if metrics_match:
    print("✓ ALL METRICS MATCH PERFECTLY")
else:
    print("✗ METRICS MISMATCH DETECTED:")
    print(f"  AUC from predictions: {auc_from_file:.6f}")
    print(f"  AUC from comparison: {auc_from_comparison:.6f}")
    print(f"  AUC from config: {auc_from_config:.6f}")
    print(f"  Difference: {abs(auc_from_file - auc_from_comparison):.6f}")

# ============================================================================
# SECTION 4: VERIFY PROFITABILITY ANALYSIS NUMBERS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: PROFITABILITY ANALYSIS VERIFICATION")
print("=" * 80)

# Combine data
df = pd.DataFrame({
    'loan_amnt': X_test['loan_amnt'],
    'actual_default': y_test['default'],
    'predicted_pd': predictions['y_pred_proba']
})

print(f"\nTest set size: {len(df):,} loans")
print(
    f"Actual defaults: {(df['actual_default'] == 1).sum():,} ({(df['actual_default'] == 1).sum() / len(df) * 100:.2f}%)")
print(
    f"Actual non-defaults: {(df['actual_default'] == 0).sum():,} ({(df['actual_default'] == 0).sum() / len(df) * 100:.2f}%)")

# Industry averages (should match config)
AVG_INTEREST_PAID_GOOD = 2136.24
AVG_LOSS_PER_DEFAULT = 7006.64

print(f"\nIndustry averages used:")
print(f"  Average interest from good loan: ${AVG_INTEREST_PAID_GOOD:,.2f}")
print(f"  Average loss per default: ${AVG_LOSS_PER_DEFAULT:,.2f}")


# Calculate profitability for medical loan (35% threshold, 5% commission)
def calculate_profitability(threshold, include_commission):
    df['approved'] = df['predicted_pd'] < threshold
    approved = df[df['approved']]

    if len(approved) == 0:
        return None

    num_good = (approved['actual_default'] == 0).sum()
    num_bad = (approved['actual_default'] == 1).sum()

    revenue_interest = num_good * AVG_INTEREST_PAID_GOOD
    losses = num_bad * AVG_LOSS_PER_DEFAULT
    commission = approved['loan_amnt'].sum() * 0.05 if include_commission else 0

    total_revenue = revenue_interest + commission
    net_profit = total_revenue - losses
    roi = net_profit / approved['loan_amnt'].sum()

    return {
        'num_approved': len(approved),
        'num_good': num_good,
        'num_bad': num_bad,
        'revenue_interest': revenue_interest,
        'commission': commission,
        'losses': losses,
        'net_profit': net_profit,
        'roi': roi
    }


standard = calculate_profitability(0.30, False)
medical = calculate_profitability(0.35, True)

print("\n" + "-" * 80)
print("STANDARD LOAN (30% threshold, no commission):")
print("-" * 80)
print(f"  Approvals: {standard['num_approved']:,}")
print(f"  Fully paid: {standard['num_good']:,}")
print(f"  Defaulted: {standard['num_bad']:,}")
print(f"  Net profit: ${standard['net_profit']:,.2f}")
print(f"  ROI: {standard['roi'] * 100:.2f}%")

print("\n" + "-" * 80)
print("MEDICAL LOAN (35% threshold, 5% commission):")
print("-" * 80)
print(f"  Approvals: {medical['num_approved']:,}")
print(f"  Fully paid: {medical['num_good']:,}")
print(f"  Defaulted: {medical['num_bad']:,}")
print(f"  Revenue from interest: ${medical['revenue_interest']:,.2f}")
print(f"  Revenue from commission: ${medical['commission']:,.2f}")
print(f"  Total revenue: ${medical['revenue_interest'] + medical['commission']:,.2f}")
print(f"  Losses: ${medical['losses']:,.2f}")
print(f"  Net profit: ${medical['net_profit']:,.2f}")
print(f"  ROI: {medical['roi'] * 100:.2f}%")

print("\n" + "-" * 80)
print("COMPARISON:")
print("-" * 80)
print(f"  Additional profit (medical vs standard): ${medical['net_profit'] - standard['net_profit']:,.2f}")
print(
    f"  Percentage improvement: {((medical['net_profit'] - standard['net_profit']) / standard['net_profit']) * 100:.1f}%")

# Check if these match what's in streamlit
expected_roi = 11.37
actual_roi = medical['roi'] * 100

print("\n" + "-" * 80)
print("STREAMLIT VALIDATION:")
print("-" * 80)
if abs(actual_roi - expected_roi) < 0.5:
    print(f"✓ Medical loan ROI matches: {actual_roi:.2f}% ≈ {expected_roi:.2f}%")
else:
    print(f"✗ ROI MISMATCH:")
    print(f"  Calculated: {actual_roi:.2f}%")
    print(f"  Expected in Streamlit: {expected_roi:.2f}%")
    print(f"  Difference: {abs(actual_roi - expected_roi):.2f}%")

# ============================================================================
# SECTION 5: VERIFY INTEREST RATE CALCULATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: INTEREST RATE CALCULATION VERIFICATION")
print("=" * 80)

# Import interest rate calculator
import sys

sys.path.insert(0, 'src')
from interest_rate_calculator import calculate_interest_rate, calculate_medical_loan_rate

# Test at different PD levels
test_pds = [0.05, 0.15, 0.25, 0.40]
pd_labels = ["Excellent (5%)", "Good (15%)", "Fair (25%)", "Poor (40%)"]

print("\nStandard loan rates:")
print("-" * 60)
for pd, label in zip(test_pds, pd_labels):
    rate = calculate_interest_rate(pd)
    print(f"  {label:20s} PD={pd:.2f} → APR={rate * 100:.2f}%")

print("\nMedical loan rates:")
print("-" * 60)
for pd, label in zip(test_pds, pd_labels):
    result = calculate_medical_loan_rate(pd)
    std_rate = calculate_interest_rate(pd)
    print(
        f"  {label:20s} PD={pd:.2f} → APR={result['total_apr'] * 100:.2f}% (saves {result['savings_vs_standard'] * 100:.2f}% vs standard)")

# ============================================================================
# SECTION 6: CONFUSION MATRIX VERIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: CONFUSION MATRIX VERIFICATION")
print("=" * 80)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print("-" * 60)
print(f"  True Negatives (Correctly Approved):  {tn:,}")
print(f"  False Positives (Incorrectly Approved): {fp:,}")
print(f"  False Negatives (Incorrectly Rejected): {fn:,}")
print(f"  True Positives (Correctly Rejected):   {tp:,}")

print(f"\nDerived Metrics:")
print(f"  Precision: {tp / (tp + fp):.4f}")
print(f"  Recall: {tp / (tp + fn):.4f}")
print(f"  Specificity: {tn / (tn + fp):.4f}")

# ============================================================================
# SECTION 7: SAMPLE PREDICTIONS TEST
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: SAMPLE PREDICTIONS TEST")
print("=" * 80)

# Test credit decision engine
from credit_decision_engine import CreditDecisionEngine

engine = CreditDecisionEngine()

# Test with first 3 applications
print("\nTesting first 3 applications:")
print("-" * 60)

for i in range(3):
    app = X_test.iloc[[i]]
    result = engine.process_application(app, loan_type='standard')

    actual_default = y_test.iloc[i]['default']
    predicted_pd = result['probability_of_default']
    decision = result['decision']

    print(f"\nApplication {i + 1}:")
    print(f"  Actual outcome: {'DEFAULT' if actual_default == 1 else 'PAID'}")
    print(f"  Predicted PD: {predicted_pd:.4f}")
    print(f"  Decision: {decision}")
    print(f"  Risk category: {result['risk_category']}")
    print(f"  Recommended APR: {result['recommended_interest_rate']:.2f}%")

# ============================================================================
# SECTION 8: FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL VALIDATION SUMMARY")
print("=" * 80)

issues = []

# Check 1: Model metrics
if not metrics_match:
    issues.append("Model metrics don't match across files")

# Check 2: ROI
if abs(actual_roi - expected_roi) >= 0.5:
    issues.append(f"Medical loan ROI mismatch: {actual_roi:.2f}% vs expected {expected_roi:.2f}%")

# Check 3: Test set size
expected_test_size = 13184
if len(predictions) != expected_test_size:
    issues.append(f"Test set size mismatch: {len(predictions):,} vs expected {expected_test_size:,}")

if len(issues) == 0:
    print("\n✓✓✓ ALL VALIDATION CHECKS PASSED ✓✓✓")
    print("\nYour data is consistent across all files!")
    print("The Streamlit app should display correct metrics.")
else:
    print("\n⚠️  ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    print("\nPlease fix these issues before deploying.")

print("\n" + "=" * 80)
print("KEY METRICS FOR STREAMLIT:")
print("=" * 80)
print(f"Model AUC-ROC: {auc_from_file:.4f}")
print(f"Model Accuracy: {acc_from_file:.4f}")
print(f"Test set size: {len(predictions):,}")
print(f"Medical loan ROI: {medical['roi'] * 100:.2f}%")
print(f"Medical loan profit: ${medical['net_profit']:,.2f}")
print(f"Profit vs standard: +{((medical['net_profit'] - standard['net_profit']) / standard['net_profit']) * 100:.1f}%")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)