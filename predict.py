"""
Kestrel Home — Returns Risk (Variant A)
Step 4: Score the real test set (test_unlabelled.csv) and write predictions.csv
in the exact shape of sample_submission.csv.
"""
import pandas as pd
import joblib
from train_model import build_matrix

def main():
    bundle = joblib.load("model.joblib")
    model, encoders = bundle["model"], bundle["encoders"]

    test = pd.read_csv("data/test_clean.csv")
    X_test, _ = build_matrix(test, encoders=encoders, fit=False)
    scores = model.predict_proba(X_test)[:, 1]

    out = pd.DataFrame({"order_id": test["order_id"], "score": scores})

    sample = pd.read_csv("data/sample_submission.csv")
    assert set(out["order_id"]) == set(sample["order_id"]), "order_id mismatch with sample_submission.csv!"
    out = out.set_index("order_id").loc[sample["order_id"]].reset_index()  # match exact row order

    out.to_csv("predictions.csv", index=False)
    print(f"Saved predictions.csv — {len(out)} rows")
    print(f"Score distribution: min={out['score'].min():.3f}, median={out['score'].median():.3f}, max={out['score'].max():.3f}")
    print(f"Orders scored above 30% risk: {(out['score']>0.3).sum()} ({(out['score']>0.3).mean():.1%})")


if __name__ == "__main__":
    main()
