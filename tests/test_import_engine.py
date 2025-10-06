# tests/test_import_engine.py
import csv
import runpy
import importlib
from pathlib import Path

import pandas as pd

MODULE_NAME = "decision_engine" 


def _reload_with_env(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    if MODULE_NAME in importlib.sys.modules:
        del importlib.sys.modules[MODULE_NAME]
    return importlib.import_module(MODULE_NAME)


def test_import_engine_basic():
    mod = importlib.import_module(MODULE_NAME)
    assert isinstance(mod.DEFAULT_CONFIG, dict)
    assert mod.DECISION_ACCEPTED and mod.DECISION_IN_REVIEW and mod.DECISION_REJECTED


def test_env_overrides_cover_try_block(monkeypatch):
    mod = _reload_with_env(monkeypatch, REJECT_AT=7, REVIEW_AT=3)
    assert mod.DEFAULT_CONFIG["score_to_decision"]["reject_at"] == 7
    assert mod.DEFAULT_CONFIG["score_to_decision"]["review_at"] == 3


def test_is_night_true_and_false():
    mod = importlib.import_module(MODULE_NAME)
    assert mod.is_night(23) is True
    assert mod.is_night(4) is True
    assert mod.is_night(12) is False


def test_high_amount_default_and_specific():
    mod = importlib.import_module(MODULE_NAME)
    thr = mod.DEFAULT_CONFIG["amount_thresholds"]
    assert mod.high_amount(2600, "digital", thr) is True
    assert mod.high_amount(3999.99, "unknown", thr) is False


def test_assess_row_hard_block_returns_early():
    mod = importlib.import_module(MODULE_NAME)
    row = pd.Series({"chargeback_count": 3, "ip_risk": "HIGH"})
    res = mod.assess_row(row, mod.DEFAULT_CONFIG)
    assert res["decision"] == mod.DECISION_REJECTED
    assert res["risk_score"] == 100
    assert "hard_block" in res["reasons"]


def test_assess_row_paths_review_reject_and_buffer():
    mod = importlib.import_module(MODULE_NAME)

    # IN_REVIEW
    row_review = pd.Series({
        "ip_risk": "medium",
        "email_risk": "medium",
        "device_fingerprint_risk": "low",
        "user_reputation": "new",
        "hour": 23,
        "bin_country": "MX", "ip_country": "US",
        "amount_mxn": 2600, "product_type": "digital",
        "latency_ms": 1000,
        "customer_txn_30d": 0,
    })
    r1 = mod.assess_row(row_review, mod.DEFAULT_CONFIG)
    assert r1["decision"] == mod.DECISION_IN_REVIEW
    for key in ["ip_risk", "email_risk", "night_hour", "geo_mismatch", "high_amount"]:
        assert key in r1["reasons"]

    # REJECTED
    row_reject = pd.Series({
        "ip_risk": "high",
        "email_risk": "high",
        "device_fingerprint_risk": "high",
        "user_reputation": "high_risk",
        "hour": 2,
        "bin_country": "BR", "ip_country": "MX",
        "amount_mxn": 10000, "product_type": "physical",
        "latency_ms": 5000,
        "customer_txn_30d": 0,
    })
    r2 = mod.assess_row(row_reject, mod.DEFAULT_CONFIG)
    assert r2["decision"] == mod.DECISION_REJECTED

    # ACCEPTED + cubre frequency_buffer (-1) porque score > 0 antes del buffer
    row_ok = pd.Series({
        "ip_risk": "medium",        
        "email_risk": "medium",           
        "device_fingerprint_risk": "low",   
        "user_reputation": "trusted", 
        "hour": 12,
        "bin_country": "MX", "ip_country": "MX",
        "amount_mxn": 100, "product_type": "digital",
        "latency_ms": 10,
        "customer_txn_30d": 5, 
    })
    r3 = mod.assess_row(row_ok, mod.DEFAULT_CONFIG)
    assert r3["decision"] == mod.DECISION_ACCEPTED
    assert "frequency_buffer" in r3["reasons"]


def test_run_writes_csv_and_returns_df(tmp_path):
    mod = importlib.import_module(MODULE_NAME)
    inp = tmp_path / "in.csv"
    outp = tmp_path / "out.csv"

    rows = [
        {"chargeback_count": 3, "ip_risk": "high", "amount_mxn": 0, "product_type": "digital"},
        {"chargeback_count": 0, "ip_risk": "low", "email_risk": "medium", "device_fingerprint_risk": "low",
         "user_reputation": "new", "hour": 23, "bin_country": "MX", "ip_country": "US",
         "amount_mxn": 2600, "product_type": "digital", "latency_ms": 0, "customer_txn_30d": 0},
    ]
    with inp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    df_out = mod.run(str(inp), str(outp))
    assert {"decision", "risk_score", "reasons"}.issubset(df_out.columns)
    assert outp.exists()
    assert int(df_out.iloc[0]["risk_score"]) == 100


def test_main_covers_argparse_and___main__(tmp_path, monkeypatch, capsys):
    mod = importlib.import_module(MODULE_NAME)

    # preparamos input y args
    inp = tmp_path / "tx.csv"
    pd.DataFrame([{"ip_risk": "low"}]).to_csv(inp, index=False)
    outp = tmp_path / "decisions.csv"

    import sys
    argv_bkp = sys.argv[:]
    sys.argv = ["prog", "--input", str(inp), "--output", str(outp)]
    try:
        # ejecuta el bloque if __name__ == "__main__"
        runpy.run_module(MODULE_NAME, run_name="__main__")
    finally:
        sys.argv = argv_bkp

    assert outp.exists()
    printed = capsys.readouterr().out.lower()
    assert "decision" in printed