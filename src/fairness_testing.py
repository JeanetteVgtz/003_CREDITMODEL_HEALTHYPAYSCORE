"""
Fairness Testing Module
Test for demographic bias in credit decisions
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

from config import PROCESSED_DATA_DIR


class FairnessTester:
    """
    Test model fairness across protected groups
    """

    def __init__(self):
        self.results = {}

    def load_data_with_demographics(self):
        """
        Load test data and extract demographic information
        Note: Lending Club data doesn't have explicit demographics,
        so we'll use proxies like home_ownership and loan purpose
        """
        X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
        predictions = pd.read_csv(PROCESSED_DATA_DIR / "pd_test_predictions.csv")

        # Create demographic groups from available features
        # Home ownership as proxy for socioeconomic status
        X_test['group_home'] = X_test[['home_ownership_RENT',
                                       'home_ownership_MORTGAGE',
                                       'home_ownership_OWN']].idxmax(axis=1)
        X_test['group_home'] = X_test['group_home'].str.replace('home_ownership_', '')

        # FICO score ranges as proxy for credit access
        X_test['group_fico'] = pd.cut(X_test['fico_range_low'],
                                      bins=[0, 660, 720, 850],
                                      labels=['Fair/Poor', 'Good', 'Excellent'])

        # Income levels
        X_test['group_income'] = pd.cut(X_test['annual_inc'],
                                        bins=[0, 50000, 100000, np.inf],
                                        labels=['Low', 'Medium', 'High'])

        # Merge with predictions
        data = X_test.copy()
        data['y_true'] = predictions['y_test'].values
        data['y_pred'] = predictions['y_pred'].values
        data['y_pred_proba'] = predictions['y_pred_proba'].values

        return data

    def calculate_group_metrics(self, data: pd.DataFrame,
                                group_col: str) -> pd.DataFrame:
        """
        Calculate metrics for each group

        Returns:
            DataFrame with metrics per group
        """
        results = []

        for group in data[group_col].unique():
            if pd.isna(group):
                continue

            group_data = data[data[group_col] == group]

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(
                group_data['y_true'],
                group_data['y_pred']
            ).ravel()

            # Calculate rates
            total = len(group_data)
            positive_rate = (tp + fp) / total if total > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            approval_rate = group_data['y_pred'].mean()
            avg_predicted_risk = group_data['y_pred_proba'].mean()

            results.append({
                'group': group,
                'count': total,
                'approval_rate': approval_rate,
                'avg_predicted_risk': avg_predicted_risk,
                'true_positive_rate': tpr,
                'false_positive_rate': fpr,
                'positive_prediction_rate': positive_rate
            })

        return pd.DataFrame(results)

    def test_demographic_parity(self, group_metrics: pd.DataFrame) -> Dict:
        """
        Test if approval rates are similar across groups
        Demographic Parity: P(Y_pred=1 | Group=A) ≈ P(Y_pred=1 | Group=B)
        """
        approval_rates = group_metrics['positive_prediction_rate'].values
        max_rate = approval_rates.max()
        min_rate = approval_rates.min()
        disparity = max_rate - min_rate

        return {
            'metric': 'Demographic Parity',
            'max_rate': max_rate,
            'min_rate': min_rate,
            'disparity': disparity,
            'is_fair': disparity < 0.10,  # Less than 10% difference
            'threshold': 0.10
        }

    def test_equal_opportunity(self, group_metrics: pd.DataFrame) -> Dict:
        """
        Test if True Positive Rates are similar across groups
        Equal Opportunity: P(Y_pred=1 | Y_true=1, Group=A) ≈ P(Y_pred=1 | Y_true=1, Group=B)
        """
        tpr_values = group_metrics['true_positive_rate'].values
        max_tpr = tpr_values.max()
        min_tpr = tpr_values.min()
        disparity = max_tpr - min_tpr

        return {
            'metric': 'Equal Opportunity',
            'max_tpr': max_tpr,
            'min_tpr': min_tpr,
            'disparity': disparity,
            'is_fair': disparity < 0.10,
            'threshold': 0.10
        }

    def test_equalized_odds(self, group_metrics: pd.DataFrame) -> Dict:
        """
        Test if both TPR and FPR are similar across groups
        """
        tpr_disparity = group_metrics['true_positive_rate'].max() - \
                        group_metrics['true_positive_rate'].min()
        fpr_disparity = group_metrics['false_positive_rate'].max() - \
                        group_metrics['false_positive_rate'].min()

        return {
            'metric': 'Equalized Odds',
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'is_fair': (tpr_disparity < 0.10) and (fpr_disparity < 0.10),
            'threshold': 0.10
        }

    def run_full_analysis(self) -> Dict:
        """
        Run complete fairness analysis
        """
        print("Loading data...")
        data = self.load_data_with_demographics()

        results = {}

        for group_col, group_name in [('group_home', 'Home Ownership'),
                                      ('group_fico', 'Credit Score Range'),
                                      ('group_income', 'Income Level')]:
            print(f"\nAnalyzing: {group_name}")
            print("=" * 60)

            metrics = self.calculate_group_metrics(data, group_col)

            print("\nGroup Metrics:")
            print(metrics.to_string(index=False))

            # Test fairness criteria
            demo_parity = self.test_demographic_parity(metrics)
            equal_opp = self.test_equal_opportunity(metrics)
            equal_odds = self.test_equalized_odds(metrics)

            print(f"\nDemographic Parity: {'PASS' if demo_parity['is_fair'] else 'FAIL'}")
            print(f"  Disparity: {demo_parity['disparity']:.4f} (threshold: {demo_parity['threshold']})")

            print(f"\nEqual Opportunity: {'PASS' if equal_opp['is_fair'] else 'FAIL'}")
            print(f"  Disparity: {equal_opp['disparity']:.4f} (threshold: {equal_opp['threshold']})")

            print(f"\nEqualized Odds: {'PASS' if equal_odds['is_fair'] else 'FAIL'}")
            print(f"  TPR Disparity: {equal_odds['tpr_disparity']:.4f}")
            print(f"  FPR Disparity: {equal_odds['fpr_disparity']:.4f}")

            results[group_name] = {
                'metrics': metrics,
                'demographic_parity': demo_parity,
                'equal_opportunity': equal_opp,
                'equalized_odds': equal_odds
            }

        self.results = results
        return results

    def plot_fairness_metrics(self, save_path: str = None):
        """
        Visualize fairness metrics
        """
        if not self.results:
            raise ValueError("Run analysis first")

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        for idx, (group_name, data) in enumerate(self.results.items()):
            metrics_df = data['metrics']

            # Approval rates
            ax1 = axes[idx, 0]
            ax1.bar(metrics_df['group'], metrics_df['approval_rate'], color='steelblue')
            ax1.set_title(f'{group_name}: Approval Rates')
            ax1.set_ylabel('Approval Rate')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)

            # True Positive Rates
            ax2 = axes[idx, 1]
            ax2.bar(metrics_df['group'], metrics_df['true_positive_rate'], color='green')
            ax2.set_title(f'{group_name}: True Positive Rates')
            ax2.set_ylabel('TPR')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


if __name__ == "__main__":
    tester = FairnessTester()
    results = tester.run_full_analysis()

    # Save plots
    tester.plot_fairness_metrics(save_path="reports/fairness_analysis.png")

    # Save detailed results
    summary = []
    for group_name, data in results.items():
        summary.append({
            'group': group_name,
            'demographic_parity': 'PASS' if data['demographic_parity']['is_fair'] else 'FAIL',
            'equal_opportunity': 'PASS' if data['equal_opportunity']['is_fair'] else 'FAIL',
            'equalized_odds': 'PASS' if data['equalized_odds']['is_fair'] else 'FAIL'
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(PROCESSED_DATA_DIR / "fairness_test_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("FAIRNESS TESTING COMPLETE")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("\nResults saved to data/processed/fairness_test_summary.csv")
    print("Plot saved to reports/fairness_analysis.png")