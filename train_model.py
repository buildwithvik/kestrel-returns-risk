"""
Kestrel Home — Returns Risk (Variant A)
Step 2: Feature engineering + model training.

Validation strategy: TIME-BASED split, not random. The model will be used
on future orders it has never seen, so we validate on the most recent 15%
of train by order_placed_at -- this simulates real deployment and avoids
an optimistic random-split score.

Only uses features genuinely available at dispatch time (see clean_data.py
for the leakage columns that were dropped).
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

FEATURES_NUM = [
    "discount_pct", "qty", "order_value_inr", "promised_delivery_days",
    "customer_prior_orders", "customer_prior_returns", "default_pincode",
    "is_gift", "list_price_inr", "warranty_months",
]
FEATURES_CAT = [
    "sales_channel", "payment_mode", "source", "city", "state",
    "shield_member", "family",
]


def add_derived_features(df):
    df = df.copy()
    df["order_placed_at"] = pd.to_datetime(df["order_placed_at"])
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["launch_date"] = pd.to_datetime(df["launch_date"])

    df["order_month"] = df["order_placed_at"].dt.month
    df["order_dow"] = df["order_placed_at"].dt.dayofweek
    df["customer_tenure_days"] = (df["order_placed_at"] - df["signup_date"]).dt.days.clip(lower=0)
    df["product_age_days"] = (df["order_placed_at"] - df["launch_date"]).dt.days.clip(lower=0)
    df["prior_return_rate"] = df["customer_prior_returns"] / df["customer_prior_orders"].replace(0, np.nan)
    df["prior_return_rate"] = df["prior_return_rate"].fillna(0)
    df["is_new_customer"] = (df["customer_prior_orders"] == 0).astype(int)
    return df


ALL_FEATURES = FEATURES_NUM + FEATURES_CAT + [
    "order_month", "order_dow", "customer_tenure_days", "product_age_days",
    "prior_return_rate", "is_new_customer",
]


def build_matrix(df, encoders=None, fit=False):
    df = add_derived_features(df)
    X = df[FEATURES_NUM + ["order_month", "order_dow", "customer_tenure_days",
                            "product_age_days", "prior_return_rate", "is_new_customer"]].copy()

    if encoders is None:
        encoders = {}
    for col in FEATURES_CAT:
        if fit:
            cats = df[col].astype("category")
            encoders[col] = cats.cat.categories
        X[col] = pd.Categorical(df[col], categories=encoders.get(col)).codes

    return X, encoders


def main():
    train = pd.read_csv("data/train_clean.csv")
    train["order_placed_at"] = pd.to_datetime(train["order_placed_at"])
    train = train.sort_values("order_placed_at").reset_index(drop=True)

    split_idx = int(len(train) * 0.85)
    tr, val = train.iloc[:split_idx], train.iloc[split_idx:]
    print(f"Train: {len(tr)} orders ({tr['order_placed_at'].min().date()} to {tr['order_placed_at'].max().date()})")
    print(f"Val:   {len(val)} orders ({val['order_placed_at'].min().date()} to {val['order_placed_at'].max().date()})")

    X_tr, encoders = build_matrix(tr, fit=True)
    y_tr = tr["returned"]
    X_val, _ = build_matrix(val, encoders=encoders, fit=False)
    y_val = val["returned"]

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        categorical_features=None,  # already integer-encoded above
        class_weight="balanced", random_state=42,
    )
    model.fit(X_tr, y_tr)

    val_scores = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_scores)
    ap = average_precision_score(y_val, val_scores)
    print(f"\nValidation ROC-AUC: {auc:.3f}")
    print(f"Validation Avg Precision (PR-AUC): {ap:.3f}  (baseline/random = {y_val.mean():.3f})")

    val_out = val[["order_id", "returned"]].copy()
    val_out["score"] = val_scores
    val_out.to_csv("data/val_scored.csv", index=False)
    print("\nSaved data/val_scored.csv (for evidence/evaluation)")

    # --- Final model: retrain on ALL of train (val included) for real predictions ---
    X_all, encoders_final = build_matrix(train, fit=True)
    y_all = train["returned"]
    final_model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        categorical_features=None, class_weight="balanced", random_state=42,
    )
    final_model.fit(X_all, y_all)
    joblib.dump({"model": final_model, "encoders": encoders_final, "features": ALL_FEATURES}, "model.joblib")
    print("Saved model.joblib (trained on ALL historical data, for real predictions)")


if __name__ == "__main__":
    main()
