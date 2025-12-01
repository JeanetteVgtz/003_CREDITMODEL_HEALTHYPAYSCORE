"""
Model Evaluation Module
Comprehensive evaluation metrics and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve,
    average_precision_score
)
from typing import Tuple, Dict
import pickle

from src.config import (
    BEST_MODEL_PATH, SCALER_PATH, PREDICTIONS_PATH,
    MODEL_COMPARISON_PATH, COLOR_PALETTE, FIGURE_SIZE
)


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization
    """

    def __init__(self):
        self.predictions_df = None
        self.model_comparison_df = None
        self._load_data()

    def _load_data(self):
        """Load predictions and model comparison data"""
        if PREDICTIONS_PATH.exists():
            self.predictions_df = pd.read_csv(PREDICTIONS_PATH)
        if MODEL_COMPARISON_PATH.exists():
            self.model_comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)

    def calculate_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                          threshold: float = 0.5) -> Dict:
        """
        Calculate comprehensive evaluation metrics

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)

        # ROC-AUC
        auc_roc = roc_auc_score(y_true, y_pred_proba)

        # Precision-Recall
        avg_precision = average_precision_score(y_true, y_pred_proba)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Derived metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'auc_roc': auc_roc,
            'avg_precision': avg_precision,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       save_path: str = None) -> plt.Figure:
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})',
                color=COLOR_PALETTE['primary'], linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                              threshold: float = 0.5, save_path: str = None) -> plt.Figure:
        """Plot confusion matrix"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(['No Default', 'Default'])
        ax.set_yticklabels(['No Default', 'Default'])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_model_comparison(self, save_path: str = None) -> plt.Figure:
        """Plot comparison of all models"""
        if self.model_comparison_df is None:
            raise ValueError("No model comparison data available")

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        df_sorted = self.model_comparison_df.sort_values('AUC-ROC', ascending=True)
        colors = [COLOR_PALETTE['primary'] if model == 'Neural Network'
                  else COLOR_PALETTE['secondary'] for model in df_sorted['Model']]

        ax.barh(df_sorted['Model'], df_sorted['AUC-ROC'], color=colors)
        ax.set_xlabel('AUC-ROC Score')
        ax.set_title('Model Performance Comparison')
        ax.axvline(x=0.79, color='red', linestyle='--', alpha=0.5, label='Target: 0.79')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def generate_evaluation_report(self, y_true: np.ndarray,
                                   y_pred_proba: np.ndarray) -> str:
        """Generate comprehensive text report"""
        metrics = self.calculate_metrics(y_true, y_pred_proba)

        report = f"""
MODEL EVALUATION REPORT
{'=' * 60}

Classification Metrics:
  - AUC-ROC Score:     {metrics['auc_roc']:.4f}
  - Average Precision: {metrics['avg_precision']:.4f}
  - Accuracy:          {metrics['accuracy']:.4f}
  - Precision:         {metrics['precision']:.4f}
  - Recall (TPR):      {metrics['recall']:.4f}
  - Specificity (TNR): {metrics['specificity']:.4f}
  - F1 Score:          {metrics['f1_score']:.4f}

Confusion Matrix:
  - True Negatives:    {metrics['true_negatives']:,}
  - False Positives:   {metrics['false_positives']:,}
  - False Negatives:   {metrics['false_negatives']:,}
  - True Positives:    {metrics['true_positives']:,}

Business Impact:
  - Loans Correctly Approved: {metrics['true_negatives']:,}
  - Loans Incorrectly Approved: {metrics['false_positives']:,}
  - Loans Correctly Rejected: {metrics['true_positives']:,}
  - Loans Incorrectly Rejected: {metrics['false_negatives']:,}
"""
        return report


if __name__ == "__main__":
    evaluator = ModelEvaluator()

    # Load test predictions
    if evaluator.predictions_df is not None:
        y_true = evaluator.predictions_df['actual'].values
        y_pred = evaluator.predictions_df['predicted_proba'].values

        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        print("Model Evaluation Metrics:")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # Generate report
        print("\n" + evaluator.generate_evaluation_report(y_true, y_pred))
    else:
        print("No predictions file found. Run model training first.")