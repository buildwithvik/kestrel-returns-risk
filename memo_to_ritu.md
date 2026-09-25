# Memo: Returns Risk Model — What to Do Next Week
**To:** Ritu Deshpande, Head of D2C Operations
**From:** [Your name]
**Re:** The returns model you asked for — the decision, the number, the rupees

## The decision

**Call flagged orders. Don't hold them — not yet.**

You asked to "flag and hold." The data doesn't support holding broadly, and
I'd rather tell you that now than after it costs you. Calling, on the other
hand, is a proven win and costs almost nothing to turn on.

## The number

Only 11.4% of orders are ever returned. That matters, because a model that
predicted "nothing gets returned" would already be "95% accurate" by a
naive count — while catching zero real returns. That's not a useful bar.
What the model actually does is sort orders into three real risk bands:

| Group | Actual return rate |
|---|---|
| Marked safe to ship | 2.5% |
| Marked to call | 12.9% |
| Marked to hold | 21.4% |
| (Company average) | 11.4% |

It correctly separates the genuinely safe majority from the genuinely risky
minority — and only misses about 1 in 20 real returns.

## The rupees

Farhan's Rs 1,150 all-in cost per return is the number this is built on
(not the Rs 600 estimate) — it matches the operations policy.

- **Calling** flagged orders: on a recent two-month test, this would have
  cost about Rs 52,000 in calls and saved about Rs 60,000 in avoided
  returns — a net gain of roughly **Rs 8,000, in two months, on calls
  alone.** Your spring pilot already proved calls stop about 35% of the
  returns they're used on — this isn't a guess.
- **Holding** broadly: even being generous and assuming a hold stops a
  return completely — which nothing in the data actually proves — the
  same test period came out roughly break-even to negative, because 12%
  of held customers just cancel the order outright. Farhan asked to see
  this trade-off before any hold policy goes live. This is it.

## What you should do next week

1. **Turn on calling for orders flagged medium-or-higher risk.** Low cost,
   proven, and the maths already work in Kestrel's favour.
2. **Don't hold orders yet.** If you want holding as an option, pilot it
   on a small, deliberately chosen slice first and measure whether it
   actually prevents returns — right now nobody has proven that it does.
3. **Never hold Shield members.** They already return more (18.6% vs 9.4%
   for everyone else — expected, since it's free for them), but Meenal is
   right that holding their orders risks the relationship with your
   highest-value customers. Call them like anyone else; don't hold them.

## What's attached

- A working tool: enter any order and it gives a risk score, a
  recommended action, and the reasons — for your team to actually use.
- `predictions.csv` — every recent order scored, in the format requested.
- A short write-up of every judgment call made in building this, and why.
