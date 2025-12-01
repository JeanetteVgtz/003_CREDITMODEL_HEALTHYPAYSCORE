"""
Model Validation Module
Cross-validation and robustness testing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle
from tensorflow import keras
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED_DATA_DIR, RANDOM_STATE


class CrossValidator:
    """
    Perform k-fold cross-validation on credit risk model
    """

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.fold_results = []

    def load_data(self):
        """Load training and validation data"""
        X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train_scaled.csv")
        y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").values.ravel()
        X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val_scaled.csv")
        y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv").values.ravel()

        # Combine for cross-validation
        X_combined = pd.concat([X_train, X_val], axis=0).values
        y_combined = np.concatenate([y_train, y_val])

        return X_combined, y_combined

    def build_model(self, input_dim: int):
        """Build neural network architecture (same as training)"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def perform_cross_validation(self) -> pd.DataFrame:
        """
        Perform stratified k-fold cross-validation

        Returns:
            DataFrame with results for each fold
        """
        print(f"Starting {self.n_folds}-fold cross-validation...")

        X, y = self.load_data()

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{self.n_folds}")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Build and train model
            model = self.build_model(X_train_fold.shape[1])

            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=50,
                batch_size=256,
                verbose=0
            )

            # Predict
            y_pred_proba = model.predict(X_val_fold, verbose=0).ravel()
            y_pred = (y_pred_proba >= 0.5).astype(int)

            # Calculate metrics
            fold_metrics = {
                'fold': fold,
                'auc_roc': roc_auc_score(y_val_fold, y_pred_proba),
                'accuracy': accuracy_score(y_val_fold, y_pred),
                'precision': precision_score(y_val_fold, y_pred, zero_division=0),
                'recall': recall_score(y_val_fold, y_pred, zero_division=0),
                'f1_score': f1_score(y_val_fold, y_pred, zero_division=0),
                'samples': len(y_val_fold)
            }

            self.fold_results.append(fold_metrics)

            print(f"  AUC-ROC: {fold_metrics['auc_roc']:.4f}")
            print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")

        results_df = pd.DataFrame(self.fold_results)

        # Calculate mean and std
        summary = {
            'fold': 'Mean ± Std',
            'auc_roc': f"{results_df['auc_roc'].mean():.4f} ± {results_df['auc_roc'].std():.4f}",
            'accuracy': f"{results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}",
            'precision': f"{results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}",
            'recall': f"{results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}",
            'f1_score': f"{results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}",
            'samples': results_df['samples'].sum()
        }

        print("\n" + "=" * 60)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        print(f"AUC-ROC:   {summary['auc_roc']}")
        print(f"Accuracy:  {summary['accuracy']}")
        print(f"Precision: {summary['precision']}")
        print(f"Recall:    {summary['recall']}")
        print(f"F1-Score:  {summary['f1_score']}")

        return results_df

    def plot_cv_results(self, save_path: str = None):
        """Plot cross-validation results"""
        if not self.fold_results:
            raise ValueError("No results to plot. Run cross-validation first.")

        df = pd.DataFrame(self.fold_results)

        metrics = ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['AUC-ROC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))

        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            values = df[metric].values
            mean_val = values.mean()
            std_val = values.std()

            ax.bar(df['fold'], values, alpha=0.7, color='steelblue')
            ax.axhline(mean_val, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_val:.4f}')
            ax.axhline(mean_val + std_val, color='orange', linestyle=':', linewidth=1,
                       label=f'±1 Std: {std_val:.4f}')
            ax.axhline(mean_val - std_val, color='orange', linestyle=':', linewidth=1)

            ax.set_xlabel('Fold')
            ax.set_ylabel(name)
            ax.set_title(f'{name} per Fold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


if __name__ == "__main__":
    validator = CrossValidator(n_folds=5)
    results = validator.perform_cross_validation()

    # Save results
    results.to_csv(PROCESSED_DATA_DIR / "cross_validation_results.csv", index=False)

    # Plot
    validator.plot_cv_results(save_path="reports/cross_validation_results.png")

    print("\nResults saved to data/processed/cross_validation_results.csv")
    print("Plot saved to reports/cross_validation_results.png")