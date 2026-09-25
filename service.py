"""
Kestrel Home — Returns Risk (Variant A)
The service: one endpoint that scores a single order and explains why,
plus a simple web page that calls it.

Runs entirely locally, no API key needed (see decisions.md for why the
"reasons" are rule-based rather than LLM-based).

Run: uvicorn service:app --reload
Then open: http://localhost:8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os

from train_model import build_matrix

app = FastAPI(title="Kestrel Returns Risk")

MODEL_PATH = "model.joblib"
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run: python clean_data.py && python train_model.py first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


# Reference stats used to explain WHY an order is risky (computed once from training data).
_REF = None


def get_reference_stats():
    global _REF
    if _REF is None:
        train = pd.read_csv("data/train_clean.csv")
        _REF = {
            "avg_return_rate": train["returned"].mean(),
            "return_rate_by_payment": train.groupby("payment_mode")["returned"].mean().to_dict(),
            "return_rate_by_family": train.groupby("family")["returned"].mean().to_dict(),
            "return_rate_shield": train.groupby("shield_member")["returned"].mean().to_dict(),
        }
    return _REF


class OrderInput(BaseModel):
    order_id: str
    order_placed_at: str
    customer_id: str
    sku: str
    sales_channel: str
    payment_mode: str
    discount_pct: int
    qty: int
    order_value_inr: float
    promised_delivery_days: int
    delivery_pincode: int
    is_gift: str
    customer_prior_orders: int
    customer_prior_returns: int
    delivery_note: str | None = None
    source: str
    city: str | None = None
    state: str | None = None
    signup_date: str | None = None
    shield_member: str | None = "N"
    family: str | None = None
    list_price_inr: float | None = None
    warranty_months: int | None = None
    launch_date: str | None = None


COST_OF_RETURN = 1150
COST_OF_CALL = 45
CALL_PREVENTS_FRACTION = 0.35
CANCEL_IF_HELD_FRACTION = 0.12


def recommend_action(score, order_value_inr, shield_member):
    call_ev = CALL_PREVENTS_FRACTION * score * COST_OF_RETURN - COST_OF_CALL
    hold_ev = score * COST_OF_RETURN - CANCEL_IF_HELD_FRACTION * order_value_inr
    if shield_member != "Y" and hold_ev > 0 and hold_ev > call_ev:
        return "HOLD"
    elif call_ev > 0:
        return "CALL"
    return "SHIP"


def explain(row: dict, score: float) -> list[str]:
    ref = get_reference_stats()
    reasons = []

    if row.get("customer_prior_returns", 0) >= 2:
        reasons.append(f"Customer has {row['customer_prior_returns']} prior returns")
    elif row.get("customer_prior_returns", 0) == 1:
        reasons.append("Customer has 1 prior return")

    if row.get("customer_prior_orders", 0) == 0:
        reasons.append("First-time customer (no order history)")

    pm = row.get("payment_mode")
    pm_rate = ref["return_rate_by_payment"].get(pm)
    if pm_rate and pm_rate > ref["avg_return_rate"] * 1.3:
        reasons.append(f"Payment mode '{pm}' has an above-average return rate ({pm_rate:.0%})")

    if row.get("shield_member") == "Y":
        reasons.append("Shield member (free returns — historically less price-sensitive about returning)")

    fam = row.get("family")
    fam_rate = ref["return_rate_by_family"].get(fam)
    if fam_rate and fam_rate > ref["avg_return_rate"] * 1.3:
        reasons.append(f"Product category '{fam}' has an above-average return rate ({fam_rate:.0%})")

    if row.get("is_gift") in ("Y", 1, "1"):
        reasons.append("Marked as a gift order")

    if row.get("delivery_pincode") == 0:
        reasons.append("No verified delivery address on file (default pincode)")

    if not reasons:
        reasons.append("No strong individual risk factors — score reflects overall order profile")

    return reasons


@app.post("/score_order")
def score_order(order: OrderInput):
    bundle = get_bundle()
    model, encoders = bundle["model"], bundle["encoders"]

    row = order.model_dump()
    df = pd.DataFrame([row])
    # fill optional fields the model expects with safe defaults if missing
    df["city"] = df["city"].fillna("Unknown")
    df["state"] = df["state"].fillna("Unknown")
    df["signup_date"] = df["signup_date"].fillna(row["order_placed_at"])
    df["shield_member"] = df["shield_member"].fillna("N")
    df["family"] = df["family"].fillna("Unknown")
    df["list_price_inr"] = df["list_price_inr"].fillna(df["order_value_inr"])
    df["warranty_months"] = df["warranty_months"].fillna(12)
    df["launch_date"] = df["launch_date"].fillna(row["order_placed_at"])
    df["default_pincode"] = (df["delivery_pincode"] == 0).astype(int)
    df["is_gift"] = (df["is_gift"] == "Y").astype(int)

    X, _ = build_matrix(df, encoders=encoders, fit=False)
    score = float(model.predict_proba(X)[:, 1][0])

    action = recommend_action(score, row["order_value_inr"], row.get("shield_member", "N"))
    reasons = explain(row, score)

    return {
        "order_id": row["order_id"],
        "score": round(score, 3),
        "risk_level": "HIGH" if score >= 0.3 else ("MEDIUM" if score >= 0.12 else "LOW"),
        "recommended_action": action,
        "reasons": reasons,
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    try:
        get_bundle()
        return {"status": "ok", "model_loaded": True}
    except HTTPException as e:
        return {"status": "model_not_found", "detail": e.detail}
