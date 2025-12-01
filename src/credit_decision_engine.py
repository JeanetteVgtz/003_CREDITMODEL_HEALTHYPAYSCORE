"""
Credit Decision Engine
Core business logic for making credit decisions
"""

import numpy as np
import pandas as pd
import pickle
from tensorflow import keras
from typing import Dict, Tuple
from pathlib import Path

from config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    get_credit_decision,
    calculate_credit_score,
    get_risk_category,
    get_max_loan_amount
)
from interest_rate_calculator import calculate_interest_rate, get_risk_tier


class CreditDecisionEngine:
    """
    Production credit decision engine
    Loads model, processes applications, returns credit decisions
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_model()
        self._load_scaler()

    def _load_model(self):
        """Load trained neural network model"""
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {BEST_MODEL_PATH}")
        self.model = keras.models.load_model(str(BEST_MODEL_PATH))

    def _load_scaler(self):
        """Load fitted scaler"""
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)

    def predict_default_probability(self, application_data) -> float:
        """
        Predict probability of default for a loan application

        Args:
            application_data: DataFrame or dict with application features

        Returns:
            Probability of default (0-1)
        """
        # Convert to DataFrame if dict
        if isinstance(application_data, dict):
            application_data = pd.DataFrame([application_data])

        # Ensure we have a DataFrame
        if not isinstance(application_data, pd.DataFrame):
            raise ValueError("application_data must be a pandas DataFrame or dict")

        # Scale features
        features_scaled = self.scaler.transform(application_data)

        # Convert to float32 to avoid TensorFlow errors
        features_scaled = features_scaled.astype(np.float32)

        # Make prediction
        prediction = self.model.predict(features_scaled, verbose=0)

        # Return probability (first element of first prediction)
        return float(prediction[0][0])

    def _get_risk_tier(self, pd_probability: float) -> str:
        """Get risk tier based on probability of default"""
        return get_risk_tier(pd_probability)

    def _calculate_credit_score(self, pd_probability: float) -> int:
        """Calculate credit score based on probability of default"""
        return calculate_credit_score(pd_probability)

    def process_application(self, application_data: Dict,
                            loan_type: str = 'standard',
                            procedure_amount: float = None) -> Dict:
        """
        Process a loan application and make credit decision.

        Args:
            application_data: Dictionary containing application features
            loan_type: 'standard' or 'medical' (default: 'standard')
            procedure_amount: Amount of medical procedure in MXN (required if loan_type='medical')

        Returns:
            Dictionary containing decision and details
        """
        from interest_rate_calculator import (
            calculate_interest_rate,
            calculate_medical_loan_rate,
            calculate_clinic_commission
        )

        # Validate loan type
        if loan_type not in ['standard', 'medical']:
            raise ValueError(f"Invalid loan_type: {loan_type}. Must be 'standard' or 'medical'")

        # Validate medical loan requirements
        if loan_type == 'medical' and procedure_amount is None:
            raise ValueError("procedure_amount is required for medical loans")

        # Get model prediction
        pd_probability = self.predict_default_probability(application_data)

        # Determine risk tier
        risk_tier = self._get_risk_tier(pd_probability)

        # Calculate credit score
        credit_score = self._calculate_credit_score(pd_probability)

        # Get appropriate thresholds based on loan type
        if loan_type == 'medical':
            from config import MEDICAL_LOAN_CONFIG
            thresholds = MEDICAL_LOAN_CONFIG['thresholds']
        else:
            from config import CREDIT_THRESHOLDS
            thresholds = CREDIT_THRESHOLDS

        # Make approval decision
        if pd_probability <= thresholds['approve']:
            decision = "Approved"
        elif pd_probability <= thresholds['review']:
            decision = "Manual Review"
        else:
            decision = "Rejected"

        # Calculate appropriate interest rate
        if loan_type == 'medical':
            rate_result = calculate_medical_loan_rate(pd_probability)
            interest_rate = rate_result['total_apr']
            interest_rate_percent = rate_result['total_apr_percent']
        else:
            interest_rate = calculate_interest_rate(pd_probability)
            interest_rate_percent = round(interest_rate * 100, 2)

        # Build base result
        result = {
            "decision": decision,
            "probability_of_default": round(pd_probability, 4),
            "credit_score": credit_score,
            "risk_category": risk_tier,
            "recommended_interest_rate": interest_rate_percent,
            "loan_type": loan_type,
            "processing_time_seconds": 0.0  # Will be calculated by caller if needed
        }

        # Add medical loan specific information
        if loan_type == 'medical':
            commission_info = calculate_clinic_commission(procedure_amount)

            result.update({
                "procedure_amount": procedure_amount,
                "clinic_commission_rate": commission_info['commission_rate_percent'],
                "clinic_commission_amount": commission_info['commission_amount'],
                "clinic_receives": commission_info['clinic_receives'],
                "lender_retains": commission_info['lender_retains'],
                "payment_method": "direct_to_provider",
                "standard_loan_rate": round(calculate_interest_rate(pd_probability) * 100, 2),
                "rate_savings_vs_standard": round(rate_result['savings_vs_standard'] * 100, 2),
                "rate_components": rate_result['components']
            })

        return result

    def batch_process(self, applications_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process multiple applications at once - OPTIMIZED

        Args:
            applications_df: DataFrame with multiple applications

        Returns:
            DataFrame with all results
        """
        # Scale all at once
        features_scaled = self.scaler.transform(applications_df)

        # Predict all at once (FAST!)
        predictions = self.model.predict(features_scaled, verbose=1, batch_size=256)
        pd_probabilities = predictions.flatten()

        # Calculate derived metrics vectorized
        credit_scores = [calculate_credit_score(pd) for pd in pd_probabilities]
        old_risk_categories = [get_risk_category(score) for score in credit_scores]
        decisions = [get_credit_decision(pd) for pd in pd_probabilities]

        # NEW: Use realistic interest rates based on PD
        interest_rates = [calculate_interest_rate(pd) for pd in pd_probabilities]
        risk_tiers = [get_risk_tier(pd) for pd in pd_probabilities]

        max_loans = [get_max_loan_amount(cat) for cat in old_risk_categories]
        confidences = [(1 - pd) * 100 for pd in pd_probabilities]

        # Build results dataframe
        results = pd.DataFrame({
            'application_id': range(len(applications_df)),
            'decision': decisions,
            'probability_of_default': [round(pd, 4) for pd in pd_probabilities],
            'credit_score': credit_scores,
            'risk_category': risk_tiers,  # Now: Excellent/Good/Fair/Poor
            'recommended_interest_rate': [round(rate * 100, 2) for rate in interest_rates],  # Now realistic
            'max_loan_amount': max_loans,
            'confidence': [round(conf, 2) for conf in confidences]
        })

        return results


if __name__ == "__main__":
    # Test the engine
    engine = CreditDecisionEngine()

    # Real test application from Lending Club data
    test_app = {'loan_amnt': 26000.0, 'funded_amnt': 26000.0, 'funded_amnt_inv': 26000.0, 'term': 0, 'int_rate': 10.99,
                'installment': 851.09, 'grade': 2, 'sub_grade': 9, 'emp_length': 0, 'annual_inc': 52000.0,
                'addr_state': 0.1379125942950767, 'dti': 19.37, 'delinq_2yrs': 0.0, 'fico_range_low': 710.0,
                'fico_range_high': 714.0, 'inq_last_6mths': 0.0, 'mths_since_last_delinq': 999.0, 'open_acc': 14.0,
                'pub_rec': 0.0, 'revol_bal': 18161.0, 'revol_util': 55.4, 'total_acc': 20.0, 'initial_list_status': 0,
                'collections_12_mths_ex_med': 0.0, 'policy_code': 1.0, 'application_type': 0, 'acc_now_delinq': 0.0,
                'tot_coll_amt': 0.0, 'tot_cur_bal': 27613.0, 'total_rev_hi_lim': 32800.0, 'acc_open_past_24mths': 2.0,
                'avg_cur_bal': 2124.0, 'bc_open_to_buy': 7584.0, 'bc_util': 66.3, 'chargeoff_within_12_mths': 0.0,
                'delinq_amnt': 0.0, 'mo_sin_old_il_acct': 187.0, 'mo_sin_old_rev_tl_op': 187.0,
                'mo_sin_rcnt_rev_tl_op': 20.0, 'mo_sin_rcnt_tl': 20.0, 'mort_acc': 0.0, 'mths_since_recent_bc': 20.0,
                'mths_since_recent_inq': 21.0, 'num_accts_ever_120_pd': 0.0, 'num_actv_bc_tl': 6.0,
                'num_actv_rev_tl': 9.0, 'num_bc_sats': 6.0, 'num_bc_tl': 6.0, 'num_il_tl': 2.0, 'num_op_rev_tl': 13.0,
                'num_rev_accts': 18.0, 'num_rev_tl_bal_gt_0': 9.0, 'num_sats': 14.0, 'num_tl_120dpd_2m': 0.0,
                'num_tl_30dpd': 0.0, 'num_tl_90g_dpd_24m': 0.0, 'num_tl_op_past_12m': 0.0, 'pct_tl_nvr_dlq': 100.0,
                'percent_bc_gt_75': 50.0, 'pub_rec_bankruptcies': 0.0, 'tax_liens': 0.0, 'tot_hi_cred_lim': 46313.0,
                'total_bal_ex_mort': 27613.0, 'total_bc_limit': 22500.0, 'total_il_high_credit_limit': 13513.0,
                'debt_settlement_flag': 0, 'has_delinq': 0, 'loan_to_income': 0.5,
                'installment_to_income': 0.1964053846153846, 'credit_history_years': 18.833675564681723,
                'home_ownership_MORTGAGE': False, 'home_ownership_OWN': False, 'home_ownership_RENT': True,
                'verification_status_Source Verified': False, 'verification_status_Verified': False,
                'purpose_credit_card': True, 'purpose_debt_consolidation': False, 'purpose_home_improvement': False,
                'purpose_house': False, 'purpose_major_purchase': False, 'purpose_medical': False,
                'purpose_moving': False, 'purpose_other': False, 'purpose_renewable_energy': False,
                'purpose_small_business': False, 'purpose_vacation': False}

    result = engine.process_application(test_app)

    print("Credit Decision Engine Test")
    print("=" * 50)
    for key, value in result.items():
        print(f"{key}: {value}")