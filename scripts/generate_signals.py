#!/usr/bin/env python3
"""AlphaSignal — Genera señales diarias (ejecutado por cron)."""
import json, joblib, subprocess, sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
scripts = Path(__file__).parent


def run_daily_update():
    print(f"[{datetime.now()}] AlphaSignal daily update...")
    # 1. Nuevos precios
    subprocess.run([sys.executable, str(scripts / "download_market_data.py")])
    # 2. Nuevas noticias
    subprocess.run([sys.executable, str(scripts / "process_news_sentiment.py")])
    # 3. GDELT
    subprocess.run([sys.executable, str(scripts / "download_gdelt.py")])
    # 4. Merge
    subprocess.run([sys.executable, str(scripts / "merge_all_features.py")])
    # 5. Generar señales (sin re-entrenar)
    if not (MODELS_DIR / "lgbm_technical.pkl").exists():
        subprocess.run([sys.executable, str(scripts / "train_all.py")])
        return

    model = joblib.load(MODELS_DIR / "lgbm_technical.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

    df = pd.read_parquet(DATA_DIR / "final_dataset.parquet")
    latest = df.sort_values(["ticker", "date"]).groupby("ticker").tail(1)
    X = latest[feature_cols].fillna(0).values
    proba = model.predict_proba(X)[:, 1]

    signals = []
    for (_, row), prob in zip(latest.iterrows(), proba):
        signal = "COMPRA" if prob > 0.60 else ("VENDE" if prob < 0.40 else "MANTIENE")
        signals.append({
            "ticker": row["ticker"], "date": str(row["date"]),
            "signal": signal, "prob_up": round(float(prob), 4),
            "confidence": round(abs(float(prob) - 0.5) * 2, 3),
        })

    with open(MODELS_DIR / "signals_cache.json", "w") as f:
        json.dump({"signals": signals, "updated": str(datetime.now())}, f, indent=2)

    print(f"✓ {len(signals)} señales actualizadas")


if __name__ == "__main__":
    run_daily_update()
