"""
Data Processing Module
Handles feature engineering and missing value imputation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import PROCESSED_DATA_DIR


class ApplicationProcessor:
    """
    Process loan applications with incomplete data
    """

    def __init__(self):
        self.feature_medians = None
        self.all_features = None
        self._load_defaults()

    def _load_defaults(self):
        """Load median values for all features"""
        medians_path = PROCESSED_DATA_DIR / "feature_medians.csv"
        if medians_path.exists():
            df = pd.read_csv(medians_path, index_col=0)
            self.feature_medians = df.iloc[:, 0].to_dict()
            self.all_features = list(self.feature_medians.keys())

    def fill_missing_features(self, partial_data: dict) -> dict:
        """
        Fill missing features with median values

        Args:
            partial_data: Dictionary with some features provided

        Returns:
            Complete dictionary with all 87 features
        """
        complete_data = self.feature_medians.copy()

        # Override with provided values
        for key, value in partial_data.items():
            if key in complete_data:
                complete_data[key] = value

        return complete_data

    def create_application_from_simple_inputs(
            self,
            annual_inc: float,
            loan_amnt: float,
            fico_score: int,
            emp_length: int,
            dti: float,
            revol_util: float,
            open_acc: int,
            total_acc: int,
            delinq_2yrs: int,
            inq_last_6mths: int,
            int_rate: float,
            term_months: int,
            home_ownership: str,
            purpose: str
    ) -> dict:
        """
        Create complete application from key inputs

        Args:
            Key loan application parameters

        Returns:
            Complete feature dictionary
        """
        # Start with defaults
        complete_app = self.feature_medians.copy()

        # Calculate derived features
        installment = self.calculate_installment(loan_amnt, int_rate, term_months)
        loan_to_income = loan_amnt / annual_inc if annual_inc > 0 else 0
        installment_to_income = (installment * 12) / annual_inc if annual_inc > 0 else 0

        # Override with provided values
        updates = {
            'annual_inc': annual_inc,
            'loan_amnt': loan_amnt,
            'funded_amnt': loan_amnt,
            'funded_amnt_inv': loan_amnt,
            'fico_range_low': fico_score - 2,
            'fico_range_high': fico_score + 2,
            'emp_length': emp_length,
            'dti': dti,
            'revol_util': revol_util,
            'open_acc': open_acc,
            'total_acc': total_acc,
            'delinq_2yrs': delinq_2yrs,
            'inq_last_6mths': inq_last_6mths,
            'int_rate': int_rate,
            'installment': installment,
            'loan_to_income': loan_to_income,
            'installment_to_income': installment_to_income,
            'term': 0 if term_months == 36 else 1,
        }

        # Handle categorical features
        # Reset all home ownership flags
        for col in complete_app.keys():
            if col.startswith('home_ownership_'):
                complete_app[col] = False

        # Set the correct home ownership
        home_col = f'home_ownership_{home_ownership.upper()}'
        if home_col in complete_app:
            complete_app[home_col] = True

        # Reset all purpose flags
        for col in complete_app.keys():
            if col.startswith('purpose_'):
                complete_app[col] = False

        # Set the correct purpose
        purpose_col = f'purpose_{purpose.lower().replace(" ", "_")}'
        if purpose_col in complete_app:
            complete_app[purpose_col] = True

        # Apply all updates
        complete_app.update(updates)

        return complete_app

    @staticmethod
    def calculate_installment(loan_amount: float, annual_rate: float,
                              term_months: int) -> float:
        """Calculate monthly installment"""
        monthly_rate = annual_rate / 100 / 12
        if monthly_rate == 0:
            return loan_amount / term_months

        numerator = loan_amount * monthly_rate * (1 + monthly_rate) ** term_months
        denominator = (1 + monthly_rate) ** term_months - 1
        return numerator / denominator

    def validate_csv_upload(self, df: pd.DataFrame) -> tuple:
        """
        Validate uploaded CSV has required features

        Returns:
            (is_valid, missing_features)
        """
        if self.all_features is None:
            return False, ["Cannot validate - medians not loaded"]

        missing = [f for f in self.all_features if f not in df.columns]
        return len(missing) == 0, missing


if __name__ == "__main__":
    processor = ApplicationProcessor()

    # Test creating application from simple inputs
    app = processor.create_application_from_simple_inputs(
        annual_inc=60000,
        loan_amnt=20000,
        fico_score=720,
        emp_length=5,
        dti=15.0,
        revol_util=30.0,
        open_acc=10,
        total_acc=15,
        delinq_2yrs=0,
        inq_last_6mths=1,
        int_rate=10.5,
        term_months=36,
        home_ownership="RENT",
        purpose="debt_consolidation"
    )

    print(f"Created application with {len(app)} features")
    print("\nSample features:")
    for i, (k, v) in enumerate(list(app.items())[:10]):
        print(f"  {k}: {v}")