# CREDIT RISK MODELING FOR PERSONAL LOANS - PROJECT SUBMISSION

**Author:** Gian Carlo Campos Sayavedra  
**Course:** Modelos de Crédito  
**Date:** Novemeber 1, 2025

---

## FOLDER CONTENTS

### 1. Final_Report.pdf
Complete project report with all analysis, methodology, and results

### 2. Diagrams/
- `business_model_diagram.png`
- `credit_risk_model_flowchart.png`

### 3. notebooks/
Six Jupyter notebooks containing all code and analysis:
- `01_data_exploration.ipynb`
- `02_preprocessing_feature_engineering.ipynb`
- `03_pd_model_development.ipynb`
- `04_lgd_ead_expected_loss.ipynb`
- `05_executive_summary.ipynb`
- `06_interest_rate_pricing.ipynb`

### 4. data/
- `loans_cleaned_final.csv` (cleaned dataset)
- `processed/` (all processed data files ready for modeling)

---

## SYSTEM REQUIREMENTS

**Python 3.8 or higher**

**Required Libraries:**
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- tensorflow

**Install all requirements:**
```bash
pip install -r requirements.txt
```

---

## HOW TO RUN THE CODE

### QUICK START (Recommended)

1. Ensure folder structure remains intact
2. Install required libraries (see above)
3. Open Jupyter Notebook or JupyterLab
4. Navigate to `notebooks/` folder
5. Run notebooks 3-6 in order:
   - `03_pd_model_development.ipynb`
   - `04_lgd_ead_expected_loss.ipynb`
   - `05_executive_summary.ipynb`
   - `06_interest_rate_pricing.ipynb`

**These notebooks are fully functional with the included processed data.**

### COMPLETE RUN (From Scratch)

If you wish to reproduce the entire analysis from raw data:

- Notebooks 1-2 require the original LendingClub raw data file
- Download from: https://www.lendingclub.com/info/download-data.action
- File: `accepted_2007_to_2018Q4.csv` (approximately 1.6GB)
- Place in: `data/raw/`
- Then run all notebooks 1-6 in order

### IMPORTANT NOTES

- All notebooks already contain outputs from our execution
- You can view all results without running any code
- Notebooks 3-6 are fully runnable with included data
- Execution time for notebooks 3-6: approximately 10-15 minutes total

---

## DATA SOURCES

### Primary Dataset
LendingClub (2007-2018)  
Source: www.lendingclub.com

---