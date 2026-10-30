import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Fix DAYS_EMPLOYED Anomalies (365243 -> Nan), added flag for later
    - Converts negative days to years
    - Compute financial leverage ratios (cresit to income, annuity to income, percentage of the total loan the borrower has to pay back each year)
    - Compute External Credit Bureau scores
    """

    df = df.copy()

    # 1. Anomaly Fix: 365243 represents pensioners/unemployed
    df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == "365243").astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})

    # 2. Convert days to years
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
    df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365.25

    # 3. Financial Burden Ratios (+ 1e-5 prevents division by zero)
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e+5)
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1e+5)

    # 4. External Credit Bureau Composite Score
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCES_MEAN'] = df[ext_sources].mean(axis=1)

    return df


def aggregate_bureau_features(bureau_path: str = "../data/bureau.csv") -> pd.DataFrame:
    """
    Aggregates 1.7M past credit bureau loans down to 1 summary row per applicant.
    """
    print("Loading and aggregating bureau.csv...")
    bureau = pd.read_csv(bureau_path)

    # 1. Main aggregations per applicant
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
        BUREAU_LOAN_COUNT=('SK_ID_BUREAU', 'count'),
        BUREAU_TOTAL_DEBT=('AMT_CREDIT_SUM_DEBT', 'sum'),
        BUREAU_TOTAL_OVERDUE=('AMT_CREDIT_SUM_OVERDUE', 'sum'),
        BUREAU_MAX_DAYS_OVERDUE=('CREDIT_DAY_OVERDUE', 'max')
    ).reset_index()

    # 2. Count active loans specifically
    active = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR')['SK_ID_BUREAU'].count()
    active = active.rename('BUREAU_ACTIVE_LOANS')

    # Merge active counts
    bureau_agg = bureau_agg.merge(active, on='SK_ID_CURR', how='left').fillna(0)
    return bureau_agg

def aggregate_previous_applications(prev_path: str = "../data/previous_application.csv") -> pd.DataFrame:
    """
    Aggregates 1.67M past Home Credit loan applications down to 1 row per applicant.
    Captures past refusal rates, approved counts, and credit amounts.
    """
    print("Loading and aggregating previous_application.csv...")
    prev = pd.read_csv(prev_path)

    # 1. Total applications per client
    prev_agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max")
    ).reset_index()

    # 2. Count refusals and approvals specifically
    refused = prev[prev["NAME_CONTRACT_STATUS"] == "Refused"].groupby("SK_ID_CURR")["SK_ID_PREV"].count().rename("PREV_REFUSED_COUNT")
    approved = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"].groupby("SK_ID_CURR")["SK_ID_PREV"].count().rename("PREV_APPROVED_COUNT")

    prev_agg = prev_agg.merge(refused, on="SK_ID_CURR", how="left").fillna(0)
    prev_agg = prev_agg.merge(approved, on="SK_ID_CURR", how="left").fillna(0)

    # 3. Refusal rate ratio (+ 1e-5 prevents division by zero)
    prev_agg["PREV_REFUSAL_RATE"] = prev_agg["PREV_REFUSED_COUNT"] / (prev_agg["PREV_APP_COUNT"] + 1e-5)

    print(f"Aggregated {len(prev):,} past applications into {len(prev_agg):,} client profiles!")
    return prev_agg


class ApplicationPreprocessor:
    """
    - Execute domain feature engineering (Anomalies and Ratios fixes)
    - Encode Categorical text columns into integers
    - Kepp track of exact column names and orders for production
    """

    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []

    def fit_transform(self, df: pd.DataFrame, target_col: str = "TARGET"):
        """It will execute on the training data, fits encoders and retuen (X, y)"""
        df_processed = create_domain_features(df)

        # Target column
        y = df_processed[target_col] if target_col in df_processed.columns else None

        # Feature columns
        exclude = [target_col, 'SK_ID_CURR']
        feature_cols = [c for c in df_processed.columns if c not in exclude]
        X = df_processed[feature_cols].copy()

        # Text/Categorical columns to encode
        object_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.to_list()

        for col in object_cols:
            le = LabelEncoder()
            X[col] = X[col].fillna("MISSING").astype(str)
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        self.feature_names = X.columns.to_list()
        return X, y

    def transform(self, df: pd.DataFrame):
        """It will execute on Test data or new user: uses fitted encoders without relearning"""
        df_processed = create_domain_features(df)

        # Get only recognized feature columns
        available = [c for c in self.feature_names if c in df_processed.columns]
        X = df_processed[available].copy()

        # Updating missing columns with NaN
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = np.nan

        # Arranging columns in correct order
        X = X[self.feature_names].copy()

        # Applying saved encoders
        for col, le in self.label_encoders.items():
            if col in X.columns:
                X[col] = X[col].fillna('MISSING').astype(str)
                # if unseen category appear, map it to
                classes = set(le.classes_)
                fallback = 'MISSING' if 'MISSING' in classes else le.classes_[0]
                X[col] = X[col].map(lambda s: s if s in classes else fallback)
                X[col] = le.transform(X[col])

        return X
