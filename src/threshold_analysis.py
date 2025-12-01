"""
Threshold Analysis Module
Analyze model performance at different decision thresholds
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)

from config import PROCESSED_DATA_DIR


class ThresholdAnalyzer:
    """
    Analyze and optimize decision thresholds
    """

    def __init__(self):
        self.predictions = pd.read_csv(PROCESSED_DATA_DIR / "pd_test_predictions.csv")
        self.y_true = self.predictions['y_test'].values
        self.y_pred_proba = self.predictions['y_pred_proba'].values

    def calculate_metrics_at_threshold(self, threshold: float) -> dict:
        """
        Calculate all metrics at a specific threshold

        Args:
            threshold: Decision threshold (0-1)

        Returns:
            Dictionary with all metrics
        """
        y_pred = (self.y_pred_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()

        total = len(self.y_true)
        approval_rate = (y_pred == 0).sum() / total  # Predicting 0 = approve

        return {
            'threshold': threshold,
            'accuracy': accuracy_score(self.y_true, y_pred),
            'precision': precision_score(self.y_true, y_pred, zero_division=0),
            'recall': recall_score(self.y_true, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_true, y_pred, zero_division=0),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'approval_rate': approval_rate,
            'default_rate_if_approved': fn / (tn + fn) if (tn + fn) > 0 else 0
        }

    def analyze_threshold_range(self, start: float = 0.1, end: float = 0.9,
                                step: float = 0.05) -> pd.DataFrame:
        """
        Analyze metrics across a range of thresholds

        Returns:
            DataFrame with metrics at each threshold
        """
        thresholds = np.arange(start, end + step, step)
        results = []

        for threshold in thresholds:
            metrics = self.calculate_metrics_at_threshold(threshold)
            results.append(metrics)

        return pd.DataFrame(results)

    def find_optimal_threshold(self, metric: str = 'f1_score') -> float:
        """
        Find threshold that maximizes a specific metric

        Args:
            metric: 'f1_score', 'precision', 'recall', or 'accuracy'

        Returns:
            Optimal threshold value
        """
        results = self.analyze_threshold_range()
        optimal_idx = results[metric].idxmax()
        optimal_threshold = results.loc[optimal_idx, 'threshold']

        return optimal_threshold

    def plot_threshold_analysis(self, save_path: str = None):
        """
        Visualize how metrics change with threshold
        """
        results = self.analyze_threshold_range()

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Precision vs Recall
        ax1 = axes[0, 0]
        ax1.plot(results['threshold'], results['precision'],
                 label='Precision', linewidth=2, marker='o', markersize=4)
        ax1.plot(results['threshold'], results['recall'],
                 label='Recall', linewidth=2, marker='s', markersize=4)
        ax1.plot(results['threshold'], results['f1_score'],
                 label='F1-Score', linewidth=2, marker='^', markersize=4)
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
        ax1.set_xlabel('Threshold', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Precision-Recall Trade-off', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Approval Rate
        ax2 = axes[0, 1]
        ax2.plot(results['threshold'], results['approval_rate'] * 100,
                 linewidth=2, color='green', marker='o', markersize=4)
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
        ax2.set_xlabel('Threshold', fontweight='bold')
        ax2.set_ylabel('Approval Rate (%)', fontweight='bold')
        ax2.set_title('Approval Rate vs Threshold', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Confusion Matrix Components
        ax3 = axes[1, 0]
        ax3.plot(results['threshold'], results['true_positives'],
                 label='True Positives (Caught Defaults)', linewidth=2, marker='o')
        ax3.plot(results['threshold'], results['false_negatives'],
                 label='False Negatives (Missed Defaults)', linewidth=2, marker='s')
        ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
        ax3.set_xlabel('Threshold', fontweight='bold')
        ax3.set_ylabel('Count', fontweight='bold')
        ax3.set_title('Default Detection', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Summary Table at key thresholds
        ax4 = axes[1, 1]
        ax4.axis('off')

        key_thresholds = [0.3, 0.5, 0.7]
        summary_data = []
        for t in key_thresholds:
            metrics = self.calculate_metrics_at_threshold(t)
            summary_data.append([
                f"{t:.1f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['approval_rate'] * 100:.1f}%",
                f"{metrics['true_positives']}/{metrics['false_negatives']}"
            ])

        table = ax4.table(cellText=summary_data,
                          colLabels=['Threshold', 'Precision', 'Recall', 'Approval', 'Caught/Missed'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.15, 0.15, 0.15, 0.15, 0.20])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax4.set_title('Comparison at Key Thresholds', fontweight='bold', pad=20)

        plt.suptitle('Threshold Analysis: Understanding Trade-offs',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def generate_recommendation_report(self) -> str:
        """
        Generate recommendation based on different use cases
        """
        results = self.analyze_threshold_range()

        # Find optimal thresholds for different objectives
        max_f1_threshold = results.loc[results['f1_score'].idxmax(), 'threshold']
        max_precision_threshold = results.loc[results['precision'].idxmax(), 'threshold']
        max_recall_threshold = results.loc[results['recall'].idxmax(), 'threshold']

        metrics_05 = self.calculate_metrics_at_threshold(0.5)
        metrics_03 = self.calculate_metrics_at_threshold(0.3)
        metrics_07 = self.calculate_metrics_at_threshold(0.7)

        report = f"""
{'=' * 80}
THRESHOLD OPTIMIZATION ANALYSIS
{'=' * 80}

CURRENT CONFIGURATION (Threshold = 0.5)
{'=' * 80}
Precision:           {metrics_05['precision']:.4f}
Recall:              {metrics_05['recall']:.4f}
F1-Score:            {metrics_05['f1_score']:.4f}
Approval Rate:       {metrics_05['approval_rate'] * 100:.1f}%
Defaults Caught:     {metrics_05['true_positives']} of {metrics_05['true_positives'] + metrics_05['false_negatives']}
Defaults Missed:     {metrics_05['false_negatives']}

ALTERNATIVE CONFIGURATIONS
{'=' * 80}

1. GROWTH-FOCUSED STRATEGY (Threshold = 0.3)
   Objective: Maximize approvals while managing risk

   Precision:        {metrics_03['precision']:.4f} (↓ more false alarms)
   Recall:           {metrics_03['recall']:.4f} (↑ catch more defaults)
   Approval Rate:    {metrics_03['approval_rate'] * 100:.1f}%
   Defaults Caught:  {metrics_03['true_positives']} (↑ {metrics_03['true_positives'] - metrics_05['true_positives']} more)

   Trade-off: Higher volume but more borderline approvals

2. CURRENT BALANCED STRATEGY (Threshold = 0.5)
   Objective: Balance precision and growth

   [Current configuration shown above]

   Trade-off: Good precision, accepts some missed defaults

3. CONSERVATIVE STRATEGY (Threshold = 0.7)
   Objective: Minimize defaults, prioritize precision

   Precision:        {metrics_07['precision']:.4f} (↑ fewer false alarms)
   Recall:           {metrics_07['recall']:.4f} (↓ miss more defaults)
   Approval Rate:    {metrics_07['approval_rate'] * 100:.1f}%
   Defaults Caught:  {metrics_07['true_positives']} (↓ {metrics_05['true_positives'] - metrics_07['true_positives']} fewer)

   Trade-off: Very high precision but lower approval volume

RECOMMENDATIONS BY BUSINESS CONTEXT
{'=' * 80}

- Fintech/Growth Stage:        Use 0.3-0.4 (maximize volume)
- Traditional Bank:             Use 0.5 (current balanced approach)
- Risk-Averse/Post-Crisis:      Use 0.6-0.7 (minimize defaults)
- Hybrid with Human Review:     Use 0.5, flag 0.3-0.5 for manual review

OPTIMAL THRESHOLDS BY METRIC
{'=' * 80}
Max F1-Score:      {max_f1_threshold:.2f}
Max Precision:     {max_precision_threshold:.2f}
Max Recall:        {max_recall_threshold:.2f}

Note: There is no universally "best" threshold. The optimal choice depends
on the institution's risk appetite, growth strategy, and cost of capital.

{'=' * 80}
"""
        return report


if __name__ == "__main__":
    analyzer = ThresholdAnalyzer()

    # Generate analysis
    print(analyzer.generate_recommendation_report())

    # Create visualization
    analyzer.plot_threshold_analysis(save_path="reports/figures/threshold_analysis.png")

    print("\nVisualization saved to: reports/figures/threshold_analysis.png")

    # Save detailed results
    results = analyzer.analyze_threshold_range()
    results.to_csv(PROCESSED_DATA_DIR / "threshold_analysis.csv", index=False)
    print("Detailed results saved to: data/processed/threshold_analysis.csv")