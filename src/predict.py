import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from src.utils import load_artifact


class CreditRiskPredictor:
    """
    Production scoring engine for new loan applicants.
    """

    def __init__(self,
                 model_path="models/credit_risk_lgb.joblib",
                 preprocessor_path="models/preprocessor.joblib",
                 threshold=0.158):
        print("Loading trained model and preprocessor...")
        self.model = load_artifact(model_path)
        self.preprocessor = load_artifact(preprocessor_path)
        self.threshold = threshold

    @staticmethod
    def probability_to_credit_score(pd_prob: float) -> int:
        """
        Scales probability of Default (0.0 to 1.0) into 300 - 850 Credit Score.
        """
        score = 850 - (pd_prob * 550.0)
        return int(np.clip(score, 300, 850))

    def get_decision(self, pd_prob: float) -> Tuple[str, str]:
        """
        Classifies risk tier and underwriting decision.
        """

        # if below 9.5% (0.095/0.157 = 0.6)
        if pd_prob < self.threshold * 0.6:
            return "APPROVE", "Low Risk"
        # if below 20.5% (0.205/0.158 = 1.29)
        elif pd_prob < self.threshold * 1.3:
            return "MANUAL REVIEW", "Moderate Risk"
        else:
            return "REJECT", "High Risk"

    def get_risk_reasons(self, applicant_row: pd.Series) -> List[str]:
        """
        Generates FCRA-compliant Adverse Action Reason codes.
        """
        reasons = []

        # Check external bureau composite score

        # The average score for safe borrowers was 0.52.
        # The average score for defaulters was 0.38.
        # The bottom 25th percentile (the lowest quartile of the population) starts at 0.35.
        ext_score = applicant_row.get('EXT_SOURCES_MEAN', np.nan)
        if pd.notnull(ext_score) and ext_score < 0.35:
            reasons.append("Low external credit bureau rating (<0.35)")

        # Check debt-to-income (DTI)
        dti = applicant_row.get('ANNUITY_TO_INCOME_RATIO', np.nan)
        if pd.notnull(dti) and dti > 0.20:
            reasons.append(f"High debt-to-income ratio ({dti:.1%})")

        # Check loan leverage
        credit_income = applicant_row.get('CREDIT_TO_INCOME_RATIO', np.nan)
        if pd.notnull(credit_income) and credit_income > 4.0:
            reasons.append(f"High loan requested relative to income ({credit_income:.1f}x annual salary)")

        # Check employment duration
        employed_years = applicant_row.get('EMPLOYED_YEARS', np.nan)
        if pd.notnull(employed_years) and employed_years < 1.0:
            reasons.append("Short employment tenure (< 1 year at current job)")

        if not reasons:
            reasons.append("Standard risk profile within baseline lending parameters")

        return reasons

    def predict(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single loan application dictionary
        """
        df_raw = pd.DataFrame([applicant_data])
        X = self.preprocessor.transform(df_raw)

        # Predict probability of default
        pd_prob = float(self.model.predict(X)[0])
        score = self.probability_to_credit_score(pd_prob)
        decision, risk_tier = self.get_decision(pd_prob)

        if decision == "APPROVE":
            reasons = ["Approved under standard lending terms. No adverse action taken."]
        else:
            reasons = self.get_risk_reasons(X.iloc[0])

        return {
            'probability_of_default': round(pd_prob, 4),
            'credit_score': score,
            'decision': decision,
            'risk_tier': risk_tier,
            'risk_reasons': reasons
        }
