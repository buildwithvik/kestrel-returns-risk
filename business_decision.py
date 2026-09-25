"""
Kestrel Home — Returns Risk (Variant A)
Step 3: Turn a risk score into an actual recommendation (call / hold / ship),
using rupee break-even math per order, not a single fixed cutoff.

Numbers used (see decisions.md for sourcing of each):
- cost of a return: Rs 1,150 (Farhan's policy figure, not Ritu's Rs 600 estimate)
- cost of a confirmation call: Rs 45
- a call prevents ~35% of the returns that would've happened (spring pilot, ops-policy.pdf)
- 12% of customers cancel if their order is held >24 hours (ops-policy.pdf)
- Shield members are never held (Meenal's warning + Ritu's own "deal with Shield later")
"""
import pandas as pd

COST_OF_RETURN = 1150
COST_OF_CALL = 45
CALL_PREVENTS_FRACTION = 0.35
CANCEL_IF_HELD_FRACTION = 0.12


def recommend_action(row):
    p = row["score"]
    order_value = row["order_value_inr"]
    is_shield = row.get("shield_member") == "Y"

    call_expected_value = CALL_PREVENTS_FRACTION * p * COST_OF_RETURN - COST_OF_CALL
    hold_expected_value = p * COST_OF_RETURN - CANCEL_IF_HELD_FRACTION * order_value

    if (not is_shield) and hold_expected_value > 0 and hold_expected_value > call_expected_value:
        return "HOLD"
    elif call_expected_value > 0:
        return "CALL"
    else:
        return "SHIP"


def main():
    val = pd.read_csv("data/val_scored.csv")
    train_clean = pd.read_csv("data/train_clean.csv")[["order_id", "order_value_inr", "shield_member"]]
    val = val.merge(train_clean, on="order_id", how="left")

    val["action"] = val.apply(recommend_action, axis=1)
    print("Action breakdown on validation set (most recent ~2 months of orders):")
    print(val["action"].value_counts())
    print()
    print("Actual return rate within each recommended action:")
    print(val.groupby("action")["returned"].agg(["mean", "count"]))

    # rough rupee impact estimate on this validation slice
    calls = val[val["action"] == "CALL"]
    holds = val[val["action"] == "HOLD"]
    call_cost = len(calls) * COST_OF_CALL
    call_savings = (CALL_PREVENTS_FRACTION * calls["returned"].sum() * COST_OF_RETURN)
    hold_cost = len(holds) * CANCEL_IF_HELD_FRACTION * holds["order_value_inr"].mean() if len(holds) else 0
    hold_savings = holds["returned"].sum() * COST_OF_RETURN if len(holds) else 0

    print(f"\nOn this ~2-month validation slice ({len(val)} orders):")
    print(f"  CALL: {len(calls)} orders, cost ~Rs {call_cost:,.0f}, est. savings ~Rs {call_savings:,.0f}")
    print(f"  HOLD: {len(holds)} orders, est. cancellation cost ~Rs {hold_cost:,.0f}, est. savings ~Rs {hold_savings:,.0f}")
    print(f"  Net estimated benefit vs no intervention: Rs {(call_savings - call_cost) + (hold_savings - hold_cost):,.0f}")

    val.to_csv("data/val_scored_actions.csv", index=False)


if __name__ == "__main__":
    main()
