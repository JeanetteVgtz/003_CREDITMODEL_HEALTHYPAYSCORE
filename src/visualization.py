"""
Visualization Module
Create professional charts for reports and presentations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from config import PROCESSED_DATA_DIR, COLOR_PALETTE, FIGURE_SIZE

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


class ReportVisualizer:
    """
    Generate publication-quality visualizations
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.predictions = pd.read_csv(PROCESSED_DATA_DIR / "pd_test_predictions.csv")
        self.model_comparison = pd.read_csv(PROCESSED_DATA_DIR / "all_models_comparison.csv")
        self.cv_results = pd.read_csv(PROCESSED_DATA_DIR / "cross_validation_results.csv")

    def plot_roc_curve_professional(self):
        """ROC curve with professional styling"""
        y_true = self.predictions['y_test']
        y_pred_proba = self.predictions['y_pred_proba']

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=3, label=f'Neural Network (AUC = {auc:.4f})',
                color='#2E86AB')

        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5000)')

        # Styling
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        # Add text box with interpretation
        textstr = 'Model Performance:\nExcellent discrimination\nbetween defaulters and\nnon-defaulters'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.60, 0.15, textstr, fontsize=11, verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("Created: roc_curve.png")

    def plot_confusion_matrix_professional(self):
        """Confusion matrix with annotations"""
        y_true = self.predictions['y_test']
        y_pred = self.predictions['y_pred']

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    cbar_kws={'label': 'Count'},
                    ax=ax, annot_kws={'size': 16, 'weight': 'bold'})

        # Labels
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
        ax.set_title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticklabels(['No Default (0)', 'Default (1)'], fontsize=12)
        ax.set_yticklabels(['No Default (0)', 'Default (1)'], fontsize=12, rotation=0)

        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = (cm[i, j] / total) * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                        ha='center', va='center', fontsize=11, color='gray')

        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("Created: confusion_matrix.png")

    def plot_model_comparison_professional(self):
        """Model comparison bar chart"""
        df = self.model_comparison.sort_values('AUC-ROC', ascending=True)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Color code - highlight best model
        colors = ['#2E86AB' if model == 'Neural Network' else '#A23B72'
                  for model in df['Model']]

        bars = ax.barh(df['Model'], df['AUC-ROC'], color=colors, alpha=0.8, edgecolor='black')

        # Add value labels
        for i, (model, auc) in enumerate(zip(df['Model'], df['AUC-ROC'])):
            ax.text(auc + 0.002, i, f'{auc:.4f}',
                    va='center', fontsize=11, fontweight='bold')

        # Styling
        ax.set_xlabel('AUC-ROC Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Model', fontsize=14, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.axvline(x=0.79, color='red', linestyle='--', linewidth=2,
                   label='Target Threshold (0.79)', alpha=0.7)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([0.78, 0.80])

        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("Created: model_comparison.png")

    def plot_cross_validation_results(self):
        """Cross-validation results visualization"""
        metrics = ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['AUC-ROC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            values = self.cv_results[metric].values
            mean_val = values.mean()
            std_val = values.std()

            # Bar plot
            bars = ax.bar(self.cv_results['fold'], values,
                          color='#2E86AB', alpha=0.7, edgecolor='black')

            # Mean line
            ax.axhline(mean_val, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_val:.4f}')

            # Std bands
            ax.axhline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5,
                       alpha=0.7)
            ax.axhline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5,
                       alpha=0.7, label=f'±1 Std: {std_val:.4f}')
            ax.fill_between(range(len(values) + 2), mean_val - std_val, mean_val + std_val,
                            alpha=0.2, color='orange')

            # Styling
            ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
            ax.set_ylabel(name, fontsize=12, fontweight='bold')
            ax.set_title(f'{name} Across Folds', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(self.cv_results['fold'])

        # Remove extra subplot
        fig.delaxes(axes[5])

        plt.suptitle('5-Fold Cross-Validation Results',
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / "cross_validation.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("Created: cross_validation.png")

    def plot_business_impact(self):
        """Business impact visualization"""
        # Calculate business metrics
        manual_time_hours = (50000 * 30) / 60  # 30 min per app
        ai_time_hours = (50000 * 3) / 3600  # 3 sec per app
        time_saved = manual_time_hours - ai_time_hours

        cost_savings = time_saved * 35  # $35/hour analyst rate

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Time comparison
        ax1 = axes[0, 0]
        processes = ['Manual Review', 'AI System']
        times = [manual_time_hours, ai_time_hours]
        colors_time = ['#A23B72', '#2E86AB']
        bars1 = ax1.bar(processes, times, color=colors_time, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Hours Required', fontsize=12, fontweight='bold')
        ax1.set_title('Processing Time: 50,000 Applications', fontsize=13, fontweight='bold')
        for bar, time in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{time:,.0f} hrs', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Cost savings
        ax2 = axes[0, 1]
        cost_data = ['Annual Cost Savings']
        savings = [cost_savings]
        bars2 = ax2.bar(cost_data, savings, color='#2E86AB', alpha=0.8, edgecolor='black', width=0.5)
        ax2.set_ylabel('USD ($)', fontsize=12, fontweight='bold')
        ax2.set_title('Annual Cost Savings', fontsize=13, fontweight='bold')
        ax2.text(0, cost_savings, f'${cost_savings:,.0f}',
                 ha='center', va='bottom', fontsize=14, fontweight='bold', color='green')
        ax2.set_ylim([0, cost_savings * 1.2])
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Throughput comparison
        ax3 = axes[1, 0]
        throughput = [50000 / manual_time_hours, 50000 / ai_time_hours]
        bars3 = ax3.bar(processes, throughput, color=colors_time, alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Applications per Hour', fontsize=12, fontweight='bold')
        ax3.set_title('Processing Throughput', fontsize=13, fontweight='bold')
        for bar, rate in zip(bars3, throughput):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{rate:,.0f}/hr', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. ROI Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        roi_text = f"""
        RETURN ON INVESTMENT SUMMARY
        {'=' * 40}

        Annual Applications:     50,000

        Manual Processing:
          • Time per app:        30 minutes
          • Total time:          {manual_time_hours:,.0f} hours
          • Annual cost:         ${manual_time_hours * 35:,.0f}

        AI System:
          • Time per app:        3 seconds
          • Total time:          {ai_time_hours:,.0f} hours
          • Annual cost:         ${ai_time_hours * 35:,.0f}

        SAVINGS:
          • Time saved:          {time_saved:,.0f} hours (99.9%)
          • Cost saved:          ${cost_savings:,.0f}
          • Throughput gain:     {(throughput[1] / throughput[0]):.0f}x faster
        """
        ax4.text(0.1, 0.5, roi_text, fontsize=11, family='monospace',
                 verticalalignment='center', bbox=dict(boxstyle='round',
                                                       facecolor='wheat', alpha=0.5))

        plt.suptitle('Business Impact Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "business_impact.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("Created: business_impact.png")

    def plot_prediction_distribution(self):
        """Distribution of predicted probabilities"""
        y_true = self.predictions['y_test']
        y_pred_proba = self.predictions['y_pred_proba']

        fig, ax = plt.subplots(figsize=(12, 6))

        # Separate by actual outcome
        defaulters = y_pred_proba[y_true == 1]
        non_defaulters = y_pred_proba[y_true == 0]

        # Plot distributions
        ax.hist(non_defaulters, bins=50, alpha=0.6, label='Actual Non-Defaulters',
                color='#2E86AB', edgecolor='black')
        ax.hist(defaulters, bins=50, alpha=0.6, label='Actual Defaulters',
                color='#A23B72', edgecolor='black')

        # Add threshold line
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2,
                   label='Decision Threshold (0.5)')

        # Styling
        ax.set_xlabel('Predicted Probability of Default', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title('Distribution of Predicted Default Probabilities',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / "prediction_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("Created: prediction_distribution.png")

    def generate_all_visualizations(self):
        """Generate all visualizations at once"""
        print("Generating professional visualizations...")
        print("=" * 60)

        self.plot_roc_curve_professional()
        self.plot_confusion_matrix_professional()
        self.plot_model_comparison_professional()
        self.plot_cross_validation_results()
        self.plot_business_impact()
        self.plot_prediction_distribution()

        print("=" * 60)
        print(f"All visualizations saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  1. roc_curve.png")
        print("  2. confusion_matrix.png")
        print("  3. model_comparison.png")
        print("  4. cross_validation.png")
        print("  5. business_impact.png")
        print("  6. prediction_distribution.png")


if __name__ == "__main__":
    visualizer = ReportVisualizer()
    visualizer.generate_all_visualizations()