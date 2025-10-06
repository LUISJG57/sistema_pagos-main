import argparse
import os
from typing import Dict, Any, List

import pandas as pd

DECISION_ACCEPTED = "ACCEPTED"
DECISION_IN_REVIEW = "IN_REVIEW"
DECISION_REJECTED = "REJECTED"

DEFAULT_CONFIG = {
    "amount_thresholds": {
        "digital": 2500,
        "physical": 6000,
        "subscription": 1500,
        "_default": 4000
    },
    "latency_ms_extreme": 2500,
    "chargeback_hard_block": 2,
    "score_weights": {
        "ip_risk": {"low": 0, "medium": 2, "high": 4},
        "email_risk": {"low": 0, "medium": 1, "high": 3, "new_domain": 2},
        "device_fingerprint_risk": {"low": 0, "medium": 2, "high": 4},
        "user_reputation": {"trusted": -2, "recurrent": -1, "new": 0, "high_risk": 4},
        "night_hour": 1,
        "geo_mismatch": 2,
        "high_amount": 2,
        "latency_extreme": 2,
        "new_user_high_amount": 2,
    },
    "score_to_decision": {
        "reject_at": 10,
        "review_at": 4
    }
}

# Optional: override thresholds via environment variables (for CI/CD / canary tuning)
try:
    _rej = os.getenv("REJECT_AT")
    _rev = os.getenv("REVIEW_AT")
    if _rej is not None:
        DEFAULT_CONFIG["score_to_decision"]["reject_at"] = int(_rej)
    if _rev is not None:
        DEFAULT_CONFIG["score_to_decision"]["review_at"] = int(_rev)
except Exception:
    pass


def is_night(hour: int) -> bool:
    return hour >= 22 or hour <= 5


def safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except Exception:
        return default


def safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def high_amount(amount: float, product_type: str, thresholds: Dict[str, Any]) -> bool:
    t = thresholds.get(product_type, thresholds.get("_default"))
    return amount >= t


def _check_chargeback_hard_block(row: pd.Series, cfg: Dict[str, Any]):
    if safe_int(row.get("chargeback_count", 0)) >= cfg["chargeback_hard_block"] and str(row.get("ip_risk", "low")).lower() == "high":
        reasons = ["hard_block:chargebacks>=2+ip_high"]
        return {"decision": DECISION_REJECTED, "risk_score": 100, "reasons": ";".join(reasons)}
    return None


def _categorical_score_and_reasons(row: pd.Series, cfg: Dict[str, Any]):
    score = 0
    reasons: List[str] = []
    for field, mapping in [("ip_risk", cfg["score_weights"]["ip_risk"]),
                           ("email_risk", cfg["score_weights"]["email_risk"]),
                           ("device_fingerprint_risk", cfg["score_weights"]["device_fingerprint_risk"])]:
        val = str(row.get(field, "low")).lower()
        add = mapping.get(val, 0)
        score += add
        if add:
            reasons.append(f"{field}:{val}(+{add})")
    return score, reasons


def _reputation_score(row: pd.Series, cfg: Dict[str, Any]):
    rep = str(row.get("user_reputation", "new")).lower()
    rep_add = cfg["score_weights"]["user_reputation"].get(rep, 0)
    reason = None
    if rep_add:
        reason = f"user_reputation:{rep}({('+' if rep_add>=0 else '')}{rep_add})"
    return rep, rep_add, reason


def _night_score(row: pd.Series, cfg: Dict[str, Any]):
    hr = safe_int(row.get("hour", 12))
    if is_night(hr):
        add = cfg["score_weights"]["night_hour"]
        return add, f"night_hour:{hr}(+{add})"
    return 0, None


def _geo_mismatch_score(row: pd.Series, cfg: Dict[str, Any]):
    bin_c = str(row.get("bin_country", "")).upper()
    ip_c = str(row.get("ip_country", "")).upper()
    if bin_c and ip_c and bin_c != ip_c:
        add = cfg["score_weights"]["geo_mismatch"]
        return add, f"geo_mismatch:{bin_c}!={ip_c}(+{add})"
    return 0, None


def _amount_score(row: pd.Series, cfg: Dict[str, Any], rep: str):
    amount = safe_float(row.get("amount_mxn", 0.0))
    ptype = str(row.get("product_type", "_default")).lower()
    if high_amount(amount, ptype, cfg["amount_thresholds"]):
        add = cfg["score_weights"]["high_amount"]
        reasons = [f"high_amount:{ptype}:{amount}(+{add})"]
        add_total = add
        if rep == "new":
            add2 = cfg["score_weights"]["new_user_high_amount"]
            add_total += add2
            reasons.append(f"new_user_high_amount(+{add2})")
        return add_total, reasons
    return 0, []


def _latency_score(row: pd.Series, cfg: Dict[str, Any]):
    lat = safe_int(row.get("latency_ms", 0))
    if lat >= cfg["latency_ms_extreme"]:
        add = cfg["score_weights"]["latency_extreme"]
        return add, f"latency_extreme:{lat}ms(+{add})"
    return 0, None


def _apply_frequency_buffer(row: pd.Series, rep: str, score: int):
    freq = safe_int(row.get("customer_txn_30d", 0))
    if rep in ("recurrent", "trusted") and freq >= 3 and score > 0:
        return -1, "frequency_buffer(-1)"
    return 0, None


def _map_score_to_decision(score: int, cfg: Dict[str, Any]) -> str:
    # Keep existing behaviour where score equal to reject_at maps to review
    if score > cfg["score_to_decision"]["reject_at"]:
        return DECISION_REJECTED
    if score >= cfg["score_to_decision"]["review_at"]:
        return DECISION_IN_REVIEW
    return DECISION_ACCEPTED


def assess_row(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Assess a transaction row and return decision, risk_score and reasons.

    Function is broken down into small helpers to make testing and coverage
    easier while preserving previous behaviour.
    """
    # Early hard block
    hb = _check_chargeback_hard_block(row, cfg)
    if hb is not None:
        return hb

    score = 0
    reasons: List[str] = []

    # Categorical
    cat_score, cat_reasons = _categorical_score_and_reasons(row, cfg)
    score += cat_score
    reasons.extend(cat_reasons)

    # Reputation
    rep, rep_add, rep_reason = _reputation_score(row, cfg)
    score += rep_add
    if rep_reason:
        reasons.append(rep_reason)

    # Night
    add, reason = _night_score(row, cfg)
    score += add
    if reason:
        reasons.append(reason)

    # Geo mismatch
    add, reason = _geo_mismatch_score(row, cfg)
    score += add
    if reason:
        reasons.append(reason)

    # Amount
    add, more_reasons = _amount_score(row, cfg, rep)
    score += add
    for r in more_reasons:
        reasons.append(r)

    # Latency
    add, reason = _latency_score(row, cfg)
    score += add
    if reason:
        reasons.append(reason)

    # Frequency buffer
    delta, fb_reason = _apply_frequency_buffer(row, rep, score)
    score += delta
    if fb_reason:
        reasons.append(fb_reason)

    decision = _map_score_to_decision(int(score), cfg)
    return {"decision": decision, "risk_score": int(score), "reasons": ";".join(reasons)}


def run(input_csv: str, output_csv: str, config: Dict[str, Any] = None) -> pd.DataFrame:
    cfg = config or DEFAULT_CONFIG
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.iterrows():
        res = assess_row(row, cfg)
        results.append(res)
    out = df.copy()
    out["decision"] = [r["decision"] for r in results]
    out["risk_score"] = [r["risk_score"] for r in results]
    out["reasons"] = [r["reasons"] for r in results]
    out.to_csv(output_csv, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, default="transactions_examples.csv", help="Path to input CSV")
    ap.add_argument("--output", required=False, default="decisions.csv", help="Path to output CSV")
    args = ap.parse_args()
    out = run(args.input, args.output)
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
