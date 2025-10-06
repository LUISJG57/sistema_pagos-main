import os
import runpy
import importlib
import pandas as pd
import pytest
from pathlib import Path
import csv

MODULE_NAME = "decision_engine"

def test_import_engine():
    import decision_engine as de
    assert isinstance(de.DEFAULT_CONFIG, dict)

def reload_with_env(monkeypatch, **env):
    """Recarga el módulo bajo prueba aplicando vars de entorno para cubrir el try/except de overrides."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # Evita cache: si ya se importó, elimínalo antes de recargar
    if MODULE_NAME in list(importlib.sys.modules):
        del importlib.sys.modules[MODULE_NAME]
    mod = importlib.import_module(MODULE_NAME)
    return mod


def test_env_overrides_change_thresholds(monkeypatch):
    mod = reload_with_env(monkeypatch, REJECT_AT="7", REVIEW_AT="3")
    assert mod.DEFAULT_CONFIG["score_to_decision"]["reject_at"] == 7
    assert mod.DEFAULT_CONFIG["score_to_decision"]["review_at"] == 3


def test_is_night_covers_true_and_false():
    mod = importlib.import_module(MODULE_NAME)
    # True (>=22) y True (<=5)
    assert mod.is_night(22) is True
    assert mod.is_night(4) is True
    # False (horas intermedias)
    assert mod.is_night(12) is False


def test_high_amount_default_and_per_type():
    mod = importlib.import_module(MODULE_NAME)
    thr = mod.DEFAULT_CONFIG["amount_thresholds"]
    # por tipo específico
    assert mod.high_amount(2600, "digital", thr) is True
    assert mod.high_amount(2499, "digital", thr) is False
    # por default
    assert mod.high_amount(4000, "unknown_x", thr) is True
    assert mod.high_amount(3999.99, "unknown_x", thr) is False


def test_assess_row_hard_block_returns_early():
    mod = importlib.import_module(MODULE_NAME)
    row = pd.Series({
        "chargeback_count": 3,
        "ip_risk": "HIGH",  # case-insensitive
    })
    res = mod.assess_row(row, mod.DEFAULT_CONFIG)
    assert res["decision"] == mod.DECISION_REJECTED
    assert res["risk_score"] == 100
    # Razón del retorno temprano
    assert "hard_block" in res["reasons"]


def test_assess_row_all_signals_paths_and_frequency_buffer():
    mod = importlib.import_module(MODULE_NAME)

    # Caso que debe quedar en IN_REVIEW (>= review_at y < reject_at)
    row_review = pd.Series({
        "ip_risk": "medium",                     # +2
        "email_risk": "new_domain",              # +2
        "device_fingerprint_risk": "low",        # +0
        "user_reputation": "new",                # +0
        "hour": 23,                              # night +1
        "bin_country": "MX", "ip_country": "US", # geo mismatch +2
        "amount_mxn": 2500.0, "product_type": "digital",  # high_amount +2
        "latency_ms": 1200,                      # no extreme
        "customer_txn_30d": 0,
    })
    res_review = mod.assess_row(row_review, mod.DEFAULT_CONFIG)
    assert res_review["decision"] == mod.DECISION_IN_REVIEW
    assert res_review["risk_score"] >= mod.DEFAULT_CONFIG["score_to_decision"]["review_at"]
    # razones variadas para cubrir branches
    for key in ["ip_risk", "email_risk", "night_hour", "geo_mismatch", "high_amount"]:
        assert key in res_review["reasons"]

    # Caso REJECTED (score alto + latency extreme)
    row_reject = pd.Series({
        "ip_risk": "high",                       # +4
        "email_risk": "high",                    # +3
        "device_fingerprint_risk": "high",       # +4
        "user_reputation": "high_risk",          # +4
        "hour": 2,                               # +1
        "bin_country": "BR", "ip_country": "MX", # +2
        "amount_mxn": 10000.0, "product_type": "physical", # +2
        "latency_ms": 5000,                      # extreme +2
        "customer_txn_30d": 0,
    })
    res_reject = mod.assess_row(row_reject, mod.DEFAULT_CONFIG)
    assert res_reject["decision"] == mod.DECISION_REJECTED
    assert res_reject["risk_score"] >= mod.DEFAULT_CONFIG["score_to_decision"]["reject_at"]

    # Caso ACCEPTED pero aplicando el buffer de frecuencia para cubrir esa rama
    row_ok = pd.Series({
        "ip_risk": "low", "email_risk": "low", "device_fingerprint_risk": "low",
        "user_reputation": "trusted",  # -2
        "hour": 12,                    # no night
        "bin_country": "MX", "ip_country": "MX",
        "amount_mxn": 100.0, "product_type": "digital",
        "latency_ms": 10,
        "customer_txn_30d": 5,         # >=3, pero score debe ser >0 para aplicar buffer
    })
    # Para forzar score > 0 antes del buffer, subimos solo un poco:
    row_ok["email_risk"] = "medium"  # +1 - 2 (trusted) -> score negativo; añadimos otra cosa:
    row_ok["ip_risk"] = "medium"     # +2 total: +3 -2 = +1, buffer lo deja en 0
    res_ok = mod.assess_row(row_ok, mod.DEFAULT_CONFIG)
    assert res_ok["decision"] == mod.DECISION_ACCEPTED
    assert "frequency_buffer" in res_ok["reasons"]


def test_run_writes_output_and_returns_df(tmp_path):
    mod = importlib.import_module(MODULE_NAME)

    # Creamos CSV de entrada con 2 filas que cubran caminos distintos
    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"

    rows = [
        {
            "chargeback_count": 3, "ip_risk": "high",  # hard block
            "amount_mxn": 0, "product_type": "digital"
        },
        {
            "chargeback_count": 0, "ip_risk": "low",
            "email_risk": "medium",
            "device_fingerprint_risk": "low",
            "user_reputation": "new",
            "hour": 23, "bin_country": "MX", "ip_country": "US",
            "amount_mxn": 2600, "product_type": "digital",
            "latency_ms": 0, "customer_txn_30d": 0
        },
    ]
    with input_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    df_out = mod.run(str(input_path), str(output_path))
    # Columnas nuevas
    assert {"decision", "risk_score", "reasons"}.issubset(df_out.columns)
    # Archivo escrito
    assert output_path.exists()
    # Primera fila es hard block con risk_score 100
    assert int(df_out.iloc[0]["risk_score"]) == 100


def test_main_cli_executes_and_prints_head(tmp_path, monkeypatch, capsys):
    """
    Cubre la función main() y la rama '__main__' ejecutando el módulo como script.
    """
    # Preparamos input/output temporales
    inp = tmp_path / "tx.csv"
    outp = tmp_path / "decisions.csv"
    pd.DataFrame([{"ip_risk": "low"}]).to_csv(inp, index=False)

    # Simulamos argumentos de línea de comandos
    monkeypatch.setenv("PYTHONPATH", str(Path.cwd()))
    monkeypatch.setenv("REVIEW_AT", "4")  # sin impacto, pero mantiene consistencia con defaults
    monkeypatch.setenv("REJECT_AT", "10")

    # run_module ejecuta el bloque if __name__ == "__main__"
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    import sys
    argv_backup = sys.argv[:]
    sys.argv = ["prog", "--input", str(inp), "--output", str(outp)]
    try:
        runpy.run_module(MODULE_NAME, run_name="__main__")
    finally:
        sys.argv = argv_backup

    # Debe imprimir un head() y haber escrito el CSV
    captured = capsys.readouterr().out
    assert "decision" in captured.lower()
    assert outp.exists()