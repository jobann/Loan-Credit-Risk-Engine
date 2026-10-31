import numpy as np
import pandas as pd

from src.features import ApplicationPreprocessor, aggregate_bureau_features, aggregate_previous_applications, \
    aggregate_installments_payments, aggregate_credit_card_balance, aggregate_pos_cash
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from src.utils import save_artifact

LGB_PARAMS = {
    'objective': 'binary',
    'metric' : 'auc',
    'boosting_type' : 'gbdt',
    'learning_rate' : 0.05,
    'num_leaves' : 31,
    'max_depth' : 7,
    'n_jobs' : -1,
    'verbosity': -1,
    'random_state': 42
}

def train_model(datapath: str="../data/application_train.csv", nrows: int=5000):
    print(f"\n--- 1. LOADING DATA ({nrows}) ---")
    df=pd.read_csv(datapath, nrows=nrows)

    # Merge bureau features!
    bureau_features = aggregate_bureau_features()
    df = df.merge(bureau_features, on='SK_ID_CURR', how='left')

    # 2. Merge previous application features (Our bank's history)
    prev_features = aggregate_previous_applications()
    df = df.merge(prev_features, on='SK_ID_CURR', how='left')

    # 3. Merge installment payment features (Repayment discipline!)
    installments_features = aggregate_installments_payments()
    df = df.merge(installments_features, on='SK_ID_CURR', how='left')

    # 4. Credit card utilization & ATM advances
    cc_features = aggregate_credit_card_balance()
    df = df.merge(cc_features, on='SK_ID_CURR', how='left')

    # 5. Point-of-sale appliance and cash loan snapshots
    pos_features = aggregate_pos_cash()
    df = df.merge(pos_features, on='SK_ID_CURR', how='left')

    print("\n--- 2. PREPROCESSING AND FEATURE ENGINEERING ---")
    preprocessor = ApplicationPreprocessor()
    X, y = preprocessor.fit_transform(df)

    print(f"Features ready: {X.shape[1]} columns. Starting 5-Fold Stratified Cross-Validation...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    models = []
    fold_scores = []

    print("\n--- 3. TRAINING FOLDS ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        #Early stopping referee (stops after 50 rounds without improvement)
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        booster = lgb.train(
            params=LGB_PARAMS,
            train_set=train_data,
            num_boost_round=1500,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Predict on validation fold using only the best trees
        val_preds = booster.predict(X_val, num_iteration=booster.best_iteration)
        oof_preds[val_idx] = val_preds

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)
        models.append(booster)

        print(f"Fold {fold} | Best Tree: {booster.best_iteration} | Validation ROC_AUC: {fold_auc:.4f}")

    # Overall Out-of-Fold Score
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n================================================================")
    print(f"Mean Fold AUC: {np.mean(fold_scores):.4f} (+/-{np.std(fold_scores):.4f})")
    print(f"Overall Out-Of-Fold (OOF) AUC: {overall_auc:.4f}")
    print(f"================================================================")

    # Save the best model and preprocessor to disk
    best_model_idx = int(np.argmax(fold_scores))
    save_artifact(models[best_model_idx], "../models/credit_risk_lgb.joblib")
    save_artifact(preprocessor, "../models/preprocessor.joblib")
    print("Saved model to 'models/credit_risk_lgb.joblib' and processor to 'models/preprocessor.joblib'")

if __name__ == "__main__":
    train_model(nrows=None)




