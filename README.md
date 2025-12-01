# HealthyPayScore - Credit Risk Assessment System

## Advanced Neural Network-Based Credit Modeling with Medical Procedure Financing Innovation

### Project Overview

This project implements a production-ready credit risk assessment system using neural networks, featuring an innovative medical procedure financing model that achieves 11.37% ROI through a dual-revenue approach (interest + 5% clinic commission).

**Key Achievement**: 341% profit improvement over baseline, 68.3% improvement over standard lending.

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JeanetteVgtz/003_CREDITMODEL_HEALTHYPAYSCORE.git
cd 003_CREDITMODEL_HEALTHYPAYSCORE
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Project Structure

```
003_CREDITMODEL_HEALTHYPAYSCORE/
│
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
│
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── config.py                 # Configuration and constants
│   ├── credit_decision_engine.py # Credit decision logic
│   ├── interest_rate_calculator.py # APR calculation
│   ├── data_processing.py        # Data preprocessing
│   ├── model_evaluation.py       # Model performance metrics
│   ├── model_validation.py       # Model validation functions
│   ├── fairness_testing.py       # Fairness analysis
│   ├── threshold_analysis.py     # Threshold optimization
│   ├── visualization.py          # Plotting functions
│   └── business_impact_real.py   # Profitability calculations
│
└── data/                         # Data files
    └── processed/                # Processed data
        ├── best_pd_model.h5      # Trained neural network
        ├── scaler.pkl            # Feature scaler
        ├── X_test.csv            # Test features (13,184 applications)
        ├── y_test.csv            # Test labels
        ├── pd_test_predictions.csv
        ├── all_models_comparison.csv
        └── feature_medians.csv
```

---

## Features

### 1. Executive Dashboard
- Real-time model performance metrics
- Medical loan ROI: 11.37%
- Profit comparison vs baseline: +341%
- Model comparison across 7 algorithms

### 2. Credit Decision Tool
- Batch processing via CSV upload
- Individual application analysis
- Risk-based APR calculation (13.88% - 27.37%)
- Instant approve/reject decisions

### 3. Model Performance
- AUC-ROC: 0.7829
- Accuracy: 83.78%
- Interactive confusion matrix
- Business metric translations

### 4. Threshold Optimization
- Interactive threshold slider
- Real-time metric recalculation
- Strategy recommendations (Growth/Balanced/Conservative)
- Confusion matrix visualization

### 5. Business Impact Analysis
- Processing efficiency: 600x faster than manual (0.002s vs 30 min)
- Real profitability calculations
- Triple impact assessment (Financial/Social/Environmental)
- Cost savings quantification

### 6. Medical Procedure Financing
Two operational modes:

**Demo Mode** (Single Application):
- Procedure amount: 5,000 - 300,000 MXN
- Loan terms: 6-24 months
- Side-by-side comparison (Standard vs Medical loan)
- APR savings calculation (approximately 1.30% average)
- Monthly payment estimates
- Clinic commission breakdown (5%)

**Batch Mode** (Portfolio Analysis):
- Process 10 to 10,000 applications
- Portfolio-level metrics
- Risk distribution analysis
- Total business impact assessment
- CSV export of detailed results

### 7. Profitability Analysis
- Baseline comparison (no model): 2.12% ROI
- Standard loan (30% threshold): 7.27% ROI
- Medical loan (35% threshold): 11.37% ROI
- Optimal threshold analysis (20%-50%)
- Revenue breakdown (interest + commission)

---

## Model Performance

### Neural Network Architecture
- **Input Layer**: 86 features
- **Hidden Layers**: 3 layers with dropout regularization
- **Output**: Probability of default (0-1)
- **Optimization**: Adam optimizer
- **Training**: 5-fold cross-validation

### Key Metrics
- **AUC-ROC**: 0.7829 (Excellent discrimination)
- **Accuracy**: 83.78%
- **Precision**: 80.84% (High confidence in rejections)
- **Recall**: 24.91% (Optimized for approvals)
- **Specificity**: 98.52% (Very few false rejections)

### Validation
- Test set: 13,184 real loan applications
- Actual defaults: 2,641 (20.03%)
- Actual fully paid: 10,543 (79.97%)
- All metrics validated against actual outcomes

---

## Business Results

### Baseline (No Model)
- **Approvals**: 13,184 (approve everyone)
- **Net Profit**: $4,017,842
- **ROI**: 2.12%

### Standard Loan Model (30% threshold)
- **Approvals**: 10,683
- **Defaults caught**: 1,298
- **Net Profit**: $10,542,564
- **ROI**: 7.27%
- **Improvement vs baseline**: +162%

### Medical Loan Model (35% threshold + 5% commission)
- **Approvals**: 11,309 (+626 more)
- **Revenue from interest**: $20,836,885
- **Revenue from commission**: $7,799,245
- **Total revenue**: $28,636,130
- **Losses**: $10,895,325
- **Net Profit**: $17,740,805
- **ROI**: 11.37%
- **Improvement vs baseline**: +341%
- **Improvement vs standard**: +68.3%

---

## Technical Details

### Data Processing
- Feature engineering: 86 derived features
- Missing value imputation using feature medians
- StandardScaler normalization
- Class imbalance handling

### Model Training
- Algorithm: Neural Network (TensorFlow/Keras)
- Cross-validation: 5-fold
- Training samples: 52,734
- Test samples: 13,184
- Hyperparameter tuning via grid search

### Fairness Testing
- Demographic parity analysis
- Equal opportunity testing
- Disparate impact ratio calculation
- Protected characteristics evaluation

### Interest Rate Calculation
Risk-based pricing formula:
```
APR = BASE_RATE + (PD × RISK_MULTIPLIER) + PROFIT_MARGIN
```
Where:
- BASE_RATE: 8%
- RISK_MULTIPLIER: 40
- PROFIT_MARGIN: 2%

Medical loan discount: -1.30% average APR reduction

---

## Data Requirements

The application requires the following files in `data/processed/`:

| File | Description | Size |
|------|-------------|------|
| `best_pd_model.h5` | Trained neural network | ~50 MB |
| `scaler.pkl` | Feature scaler | ~50 KB |
| `X_test.csv` | Test features (13,184 × 86) | ~10 MB |
| `y_test.csv` | Test labels (13,184 × 1) | ~100 KB |
| `pd_test_predictions.csv` | Model predictions | ~500 KB |
| `all_models_comparison.csv` | Model comparison results | ~5 KB |
| `feature_medians.csv` | Feature medians for imputation | ~5 KB |

---

## Troubleshooting

### Common Issues

**Error: "Model file not found"**
- Ensure `data/processed/best_pd_model.h5` exists
- Check file permissions

**Error: "Module not found"**
- Activate virtual environment
- Run `pip install -r requirements.txt`

**Streamlit app not loading**
- Check port 8501 is not in use
- Try `streamlit run streamlit_app.py --server.port 8502`

**Data file errors**
- Verify all 7 required files are in `data/processed/`
- Check CSV files are not corrupted

---

## Academic Context

This project was developed for the **Modelos de Crédito** course at **ITESO** (Instituto Tecnológico y de Estudios Superiores de Occidente) under Professor **Rodolfo Slay Ramos**.

### Project Team
- Jeanette Vazquez
- Gian Carlo
- Paulina Mejia

### Key Learning Objectives
- Credit risk modeling with neural networks
- Model validation and fairness testing
- Business impact analysis
- Interest rate pricing strategies
- Production-ready ML deployment
- Interactive data visualization

### References
1. Townson, S. (2020). "AI Can Make Bank Loans More Fair." Harvard Business Review.
2. U.S. Bureau of Labor Statistics - Salary data for cost analysis
3. McKinsey & Company - Credit processing benchmarks

---

## Technologies Used

**Built with:**
- Python 3.10+
- TensorFlow 2.15 (Neural Network)
- Streamlit 1.31 (Web Application)
- Scikit-learn 1.4.0 (Preprocessing & Metrics)
- Pandas & NumPy (Data Processing)
- Plotly, Matplotlib, Seaborn (Visualization)

