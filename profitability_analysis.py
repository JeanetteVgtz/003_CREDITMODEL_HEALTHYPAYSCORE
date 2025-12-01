"""
Real Profitability Analysis
Uses ACTUAL loan outcomes from your model's test set
"""

import pandas as pd
import numpy as np

# Load your model's predictions (this already has the actual outcomes!)
predictions = pd.read_csv('data/processed/pd_test_predictions.csv')

# Load the test features to get loan amounts
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

print("="*80)
print("REAL PROFITABILITY ANALYSIS")
print("="*80)
print(f"\nAnalyzing {len(predictions):,} loans from your model's test set")

# Combine everything
df = pd.DataFrame({
    'loan_amnt': X_test['loan_amnt'],
    'actual_default': y_test['default'],
    'predicted_pd': predictions['y_pred_proba'],
    'loan_status': ['Charged Off' if d == 1 else 'Fully Paid' for d in y_test['default']]
})

# Calculate actual outcomes
print("\n1. ACTUAL OUTCOMES IN TEST SET:")
print(f"  Fully Paid: {(df['actual_default']==0).sum():,} ({(df['actual_default']==0).sum()/len(df)*100:.1f}%)")
print(f"  Defaulted: {(df['actual_default']==1).sum():,} ({(df['actual_default']==1).sum()/len(df)*100:.1f}%)")

# Estimate financial outcomes (since we don't have total_pymnt in processed data)
# We'll use industry averages from the full dataset
AVG_INTEREST_PAID_GOOD = 2136.24  # From full dataset
AVG_LOSS_PER_DEFAULT = 7006.64    # From full dataset

print("\n2. ESTIMATED PORTFOLIO PERFORMANCE (without model):")
total_lent = df['loan_amnt'].sum()
num_good = (df['actual_default']==0).sum()
num_bad = (df['actual_default']==1).sum()

revenue_from_good = num_good * AVG_INTEREST_PAID_GOOD
losses_from_bad = num_bad * AVG_LOSS_PER_DEFAULT
net_profit_baseline = revenue_from_good - losses_from_bad

print(f"  Total lent: ${total_lent:,.2f}")
print(f"  Revenue from {num_good:,} good loans: ${revenue_from_good:,.2f}")
print(f"  Losses from {num_bad:,} defaults: ${losses_from_bad:,.2f}")
print(f"  Net profit: ${net_profit_baseline:,.2f}")
print(f"  ROI: {(net_profit_baseline/total_lent)*100:.2f}%")

def calculate_real_profitability(threshold=0.35, include_commission=True):
    """
    Calculate profitability using REAL outcomes
    """
    # Decide who to approve based on model
    df['approved'] = df['predicted_pd'] < threshold

    # Get only approved loans
    approved = df[df['approved']]

    if len(approved) == 0:
        return None

    # Calculate ACTUAL results
    num_good = (approved['actual_default']==0).sum()
    num_bad = (approved['actual_default']==1).sum()

    # Revenue from good loans
    revenue_from_interest = num_good * AVG_INTEREST_PAID_GOOD

    # Losses from defaults
    actual_losses = num_bad * AVG_LOSS_PER_DEFAULT

    # Commission (if medical loan)
    commission_revenue = 0
    if include_commission:
        commission_revenue = approved['loan_amnt'].sum() * 0.05

    # Total revenue
    total_revenue = revenue_from_interest + commission_revenue

    # Net profit
    net_profit = total_revenue - actual_losses

    return {
        'threshold': threshold,
        'num_applications': len(df),
        'num_approved': len(approved),
        'approval_rate': len(approved) / len(df),
        'num_fully_paid': num_good,
        'num_defaulted': num_bad,
        'default_rate': num_bad / len(approved),
        'total_lent': approved['loan_amnt'].sum(),
        'revenue_from_interest': revenue_from_interest,
        'commission_revenue': commission_revenue,
        'total_revenue': total_revenue,
        'actual_losses': actual_losses,
        'net_profit': net_profit,
        'profit_per_loan': net_profit / len(approved),
        'roi': net_profit / approved['loan_amnt'].sum()
    }

print("\n" + "="*80)
print("WITH YOUR MODEL:")
print("="*80)

print("\nSTANDARD LOAN (30% threshold, no commission):")
standard = calculate_real_profitability(threshold=0.30, include_commission=False)

print(f"\nApprovals:")
print(f"  Applications: {standard['num_applications']:,}")
print(f"  Approved: {standard['num_approved']:,} ({standard['approval_rate']*100:.1f}%)")
print(f"  Fully paid: {standard['num_fully_paid']:,}")
print(f"  Defaulted: {standard['num_defaulted']:,} ({standard['default_rate']*100:.1f}%)")

print(f"\nFinancials:")
print(f"  Total lent: ${standard['total_lent']:,.2f}")
print(f"  Revenue from interest: ${standard['revenue_from_interest']:,.2f}")
print(f"  Actual losses: ${standard['actual_losses']:,.2f}")
print(f"  Net profit: ${standard['net_profit']:,.2f}")
print(f"  Profit per loan: ${standard['profit_per_loan']:,.2f}")
print(f"  ROI: {standard['roi']*100:.2f}%")

print("\n" + "="*80)
print("MEDICAL LOAN (35% threshold, with 5% commission):")
medical = calculate_real_profitability(threshold=0.35, include_commission=True)

print(f"\nApprovals:")
print(f"  Applications: {medical['num_applications']:,}")
print(f"  Approved: {medical['num_approved']:,} ({medical['approval_rate']*100:.1f}%)")
print(f"  Fully paid: {medical['num_fully_paid']:,}")
print(f"  Defaulted: {medical['num_defaulted']:,} ({medical['default_rate']*100:.1f}%)")

print(f"\nFinancials:")
print(f"  Total lent: ${medical['total_lent']:,.2f}")
print(f"  Revenue from interest: ${medical['revenue_from_interest']:,.2f}")
print(f"  Revenue from commission: ${medical['commission_revenue']:,.2f}")
print(f"  Total revenue: ${medical['total_revenue']:,.2f}")
print(f"  Actual losses: ${medical['actual_losses']:,.2f}")
print(f"  Net profit: ${medical['net_profit']:,.2f}")
print(f"  Profit per loan: ${medical['profit_per_loan']:,.2f}")
print(f"  ROI: {medical['roi']*100:.2f}%")

print("\n" + "="*80)
print("COMPARISON:")
print("="*80)

print(f"\nAdditional approvals (medical vs standard): {medical['num_approved'] - standard['num_approved']:,}")
print(f"Additional profit (medical vs standard): ${medical['net_profit'] - standard['net_profit']:,.2f}")
print(f"Profit improvement: {((medical['net_profit'] - standard['net_profit'])/standard['net_profit'])*100:.1f}%")

print(f"\nProfit vs baseline (no model): ${medical['net_profit'] - net_profit_baseline:,.2f}")
print(f"Improvement: {((medical['net_profit'] - net_profit_baseline)/net_profit_baseline)*100:.1f}%")

# Find optimal threshold
print("\n" + "="*80)
print("FINDING OPTIMAL THRESHOLD:")
print("="*80)

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
results = []

for t in thresholds:
    result = calculate_real_profitability(threshold=t, include_commission=True)
    if result:
        results.append(result)

df_results = pd.DataFrame(results)

print(f"\n{'Threshold':<12}{'Approved':<12}{'Defaults':<12}{'Net Profit':<18}{'ROI':<10}")
print("-"*70)

for _, row in df_results.iterrows():
    print(f"{row['threshold']:<12.0%}{row['num_approved']:<12,.0f}"
          f"{row['num_defaulted']:<12,.0f}${row['net_profit']:<17,.0f}"
          f"{row['roi']:<10.2%}")

optimal = df_results.loc[df_results['net_profit'].idxmax()]
print(f"\nOPTIMAL THRESHOLD: {optimal['threshold']:.0%}")
print(f"  Maximum profit: ${optimal['net_profit']:,.2f}")
print(f"  ROI: {optimal['roi']*100:.2f}%")