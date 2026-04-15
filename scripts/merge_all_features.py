#!/usr/bin/env python3
"""AlphaSignal — Merge de todas las fuentes de features."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def merge_all():
    print("=== AlphaSignal — Merge Features ===")

    # Técnicas
    tech = pd.read_parquet(DATA_DIR / "technical_features.parquet")
    tech["date"] = pd.to_datetime(tech["date"]).dt.date.astype(str)
    print(f"Técnicas: {len(tech):,} filas")

    # Sentimiento (si existe)
    sentiment = None
    if (DATA_DIR / "sentiment_features.parquet").exists():
        sentiment = pd.read_parquet(DATA_DIR / "sentiment_features.parquet")
        sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.date.astype(str)
        print(f"Sentimiento: {len(sentiment):,} filas")

    # GDELT (si existe)
    gdelt = None
    if (DATA_DIR / "gdelt_features.parquet").exists():
        gdelt = pd.read_parquet(DATA_DIR / "gdelt_features.parquet")
        gdelt["date"] = pd.to_datetime(gdelt["date"]).dt.date.astype(str)
        print(f"GDELT: {len(gdelt):,} filas")

    # Merge
    merged = tech.copy()
    if sentiment is not None:
        merged = merged.merge(
            sentiment[["date", "ticker", "sentiment_daily", "sentiment_7d",
                        "sentiment_30d", "sentiment_momentum", "n_articles"]],
            on=["date", "ticker"], how="left"
        )
        merged["sentiment_daily"] = merged["sentiment_daily"].fillna(0)
        merged["sentiment_7d"] = merged["sentiment_7d"].fillna(0)
        merged["sentiment_30d"] = merged["sentiment_30d"].fillna(0)
        merged["sentiment_momentum"] = merged["sentiment_momentum"].fillna(0)
        merged["n_articles"] = merged["n_articles"].fillna(0)
    else:
        merged["sentiment_daily"] = 0.0
        merged["sentiment_7d"] = 0.0
        merged["sentiment_30d"] = 0.0
        merged["sentiment_momentum"] = 0.0
        merged["n_articles"] = 0

    if gdelt is not None:
        merged = merged.merge(
            gdelt[["date", "tension_score", "tension_7d", "tension_30d",
                   "tension_delta_1d", "extreme_event"]],
            on="date", how="left"
        )
        merged["tension_score"] = merged["tension_score"].fillna(0)
        merged["tension_7d"] = merged["tension_7d"].fillna(0)
        merged["tension_30d"] = merged["tension_30d"].fillna(0)
        merged["tension_delta_1d"] = merged["tension_delta_1d"].fillna(0)
        merged["extreme_event"] = merged["extreme_event"].fillna(0)
    else:
        merged["tension_score"] = 0.0
        merged["tension_7d"] = 0.0
        merged["tension_30d"] = 0.0
        merged["tension_delta_1d"] = 0.0
        merged["extreme_event"] = 0

    # Lags de features (1, 3, 5 días)
    lag_cols = ["ret_1d", "rsi_14", "macd", "bb_position", "sentiment_daily", "tension_score"]
    merged = merged.sort_values(["ticker", "date"])
    for col in lag_cols:
        if col in merged.columns:
            for lag in [1, 3, 5]:
                merged[f"{col}_lag{lag}"] = merged.groupby("ticker")[col].shift(lag)

    # Fillna
    num_cols = merged.select_dtypes(include=[np.number]).columns
    merged[num_cols] = merged[num_cols].fillna(0)

    out_path = DATA_DIR / "final_dataset.parquet"
    merged.to_parquet(out_path)
    print(f"✓ Dataset final: {len(merged):,} filas, {len(merged.columns)} columnas")
    return merged


if __name__ == "__main__":
    merge_all()
