"""
Kestrel Home — Returns Risk (Variant A)
Step 1: Clean train and test, fix known data bugs.

Fixes applied (see decisions.md for why):
1. Deduplicate orders re-imported from the partner-outlet feed (source=partner_feed
   duplicates a source=crm row for the same order_id).
2. Rescale order_value_inr /100 for October 2025 orders (new payment gateway
   stored values ~100x too high that month; unchecked per Tanmay's email).
3. DROP last_service_event_type and pickup_scheduled_at as model features.
   These are populated by events that happen AFTER a return process starts.
   REVERSE_PICKUP is 100% correlated with returned=1 in train, and test
   (the real dispatch-time snapshot) never contains these leaked states.
   Using them would produce a model that looks excellent offline and is
   useless in production, because this data does not exist at dispatch time.
"""
import pandas as pd
import numpy as np

DATA = "data"


def dedupe(df):
    before = len(df)
    df = df.sort_values("source").drop_duplicates("order_id", keep="first")  # 'crm' < 'partner_feed' alphabetically
    print(f"[dedupe] {before} -> {len(df)} rows ({before - len(df)} removed)")
    return df


def fix_october_scaling(df):
    dt = pd.to_datetime(df["order_placed_at"])
    is_oct25 = (dt.dt.year == 2025) & (dt.dt.month == 10)
    df.loc[is_oct25, "order_value_inr"] = df.loc[is_oct25, "order_value_inr"] / 100
    print(f"[scaling] rescaled {is_oct25.sum()} October-2025 orders /100")
    return df


def clean(path, is_train):
    df = pd.read_csv(path)
    df = dedupe(df)
    df = fix_october_scaling(df)
    df["order_placed_at"] = pd.to_datetime(df["order_placed_at"])

    # LEAKAGE: drop columns not genuinely available at dispatch time.
    # (Kept out entirely rather than "handled carefully" -- test never has
    # these leaked states, so any signal we learned from them in train
    # would not transfer.)
    leak_cols = [c for c in ["last_service_event_type", "pickup_scheduled_at"] if c in df.columns]
    df = df.drop(columns=leak_cols)
    print(f"[leakage] dropped columns: {leak_cols}")

    df["default_pincode"] = (df["delivery_pincode"] == 0).astype(int)
    df["is_gift"] = (df["is_gift"] == "Y").astype(int)

    return df


def main():
    customers = pd.read_csv(f"{DATA}/customers.csv")
    products = pd.read_csv(f"{DATA}/products.csv")

    train = clean(f"{DATA}/train.csv", is_train=True)
    test = clean(f"{DATA}/test_unlabelled.csv", is_train=False)

    for name, df in [("train", train), ("test", test)]:
        df = df.merge(customers, on="customer_id", how="left")
        df = df.merge(products, on="sku", how="left")
        df.to_csv(f"{DATA}/{name}_clean.csv", index=False)
        print(f"Saved {DATA}/{name}_clean.csv — {len(df)} rows")

    print(f"\nTrain return rate: {train['returned'].mean():.3f}")


if __name__ == "__main__":
    main()
