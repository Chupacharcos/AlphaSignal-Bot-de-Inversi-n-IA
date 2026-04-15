#!/usr/bin/env python3
"""
AlphaSignal — GDELT Events Pipeline
Descarga eventos geopolíticos de la API GDELT v2 y genera features.
"""
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERIES = [
    "war sanctions tariffs",
    "Spain economy IBEX",
    "EU Europe financial crisis",
    "United States trade tariff",
    "oil energy price",
    "bank financial collapse",
]


def fetch_gdelt_events(days_back=90):
    """Descarga eventos GDELT relevantes (sin API key)."""
    end = datetime.now()
    records = []

    for query in QUERIES:
        try:
            params = {
                "query": query,
                "mode": "artlist",
                "maxrecords": 250,
                "format": "json",
                "timespan": f"{days_back}d",
            }
            r = requests.get(GDELT_BASE, params=params, timeout=30)
            if r.status_code == 200:
                articles = r.json().get("articles", [])
                for art in articles:
                    records.append({
                        "title": art.get("title", ""),
                        "url": art.get("url", ""),
                        "seendate": art.get("seendate", ""),
                        "domain": art.get("domain", ""),
                        "query": query,
                        "tone": float(art.get("tone", "0").split(",")[0]) if art.get("tone") else 0.0,
                    })
        except Exception as e:
            print(f"Error GDELT query '{query}': {e}")

    print(f"GDELT: {len(records)} artículos descargados")
    return records


def compute_geopolitical_features(records, n_days=1500, seed=42):
    """
    Genera features geopolíticas agregadas por día.
    Si no hay datos reales, genera sintéticos.
    """
    if not records:
        print("Generando features geopolíticas sintéticas...")
        return generate_synthetic_gdelt(n_days, seed)

    df = pd.DataFrame(records)

    # Parsear fechas
    df["date"] = pd.to_datetime(df["seendate"], errors="coerce")
    df["date"] = df["date"].dt.date.astype(str)
    df = df.dropna(subset=["date"])

    # Tone: negativo = tensión, positivo = estabilidad
    # Aggregar por día
    daily = df.groupby("date").agg(
        tension_score=("tone", lambda x: -x.mean()),  # Invertido: negatividad = tensión
        n_articles=("tone", "count"),
        tone_std=("tone", "std"),
    ).reset_index()

    # Rolling tensions
    daily = daily.sort_values("date")
    daily["tension_7d"] = daily["tension_score"].rolling(7, min_periods=1).mean()
    daily["tension_30d"] = daily["tension_score"].rolling(30, min_periods=1).mean()
    daily["tension_delta_1d"] = daily["tension_score"].diff(1)
    daily["extreme_event"] = (np.abs(daily["tension_score"]) > daily["tension_score"].rolling(30).std() * 2).astype(int)

    daily.to_parquet(DATA_DIR / "gdelt_features.parquet")
    print(f"✓ GDELT features: {len(daily)} días")
    return daily


def generate_synthetic_gdelt(n_days=1500, seed=42):
    """Genera features geopolíticas sintéticas."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2019-01-02", periods=n_days, freq="B")

    # Eventos sintéticos (regímenes de tensión)
    base_tension = np.zeros(n_days)
    # COVID (2020)
    base_tension[250:400] = rng.uniform(2, 4, 150)
    # Ucrania (2022)
    base_tension[750:850] = rng.uniform(3, 5, 100)
    # Aranceles EEUU (2024)
    base_tension[1300:] = rng.uniform(1.5, 3, n_days - 1300)

    noise = rng.normal(0, 0.5, n_days)
    tension = np.clip(base_tension + noise, -3, 6)
    n_articles = rng.poisson(lam=50, size=n_days)

    df = pd.DataFrame({
        "date": [str(d.date()) for d in dates],
        "tension_score": tension,
        "n_articles": n_articles,
        "tone_std": rng.uniform(0.5, 2.0, n_days),
    })
    df["tension_7d"] = df["tension_score"].rolling(7, min_periods=1).mean()
    df["tension_30d"] = df["tension_score"].rolling(30, min_periods=1).mean()
    df["tension_delta_1d"] = df["tension_score"].diff(1).fillna(0)
    df["extreme_event"] = (np.abs(df["tension_score"]) > 3.5).astype(int)

    df.to_parquet(DATA_DIR / "gdelt_features.parquet")
    print(f"✓ GDELT sintético: {len(df):,} registros")
    return df


if __name__ == "__main__":
    print("=== AlphaSignal — GDELT Pipeline ===")
    records = fetch_gdelt_events(days_back=365)
    compute_geopolitical_features(records)
    print("\n✓ GDELT pipeline completado.")
