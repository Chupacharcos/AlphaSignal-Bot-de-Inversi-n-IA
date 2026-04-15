#!/usr/bin/env python3
"""
AlphaSignal — Pipeline de Sentimiento FinBERT
Descarga headlines RSS y aplica FinBERT para scoring de sentimiento.
Se ejecuta en batch (cron diario), NO en cada request de la API.
"""
import feedparser
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# RSS feeds financieros gratuitos
RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.expansion.com/rss/mercados.xml",
]

# Mapping empresa → ticker (IBEX35)
ENTITY_MAP = {
    "santander": "SAN.MC", "bbva": "BBVA.MC", "inditex": "ITX.MC", "zara": "ITX.MC",
    "telefonica": "TEF.MC", "telefónica": "TEF.MC", "iberdrola": "IBE.MC",
    "repsol": "REP.MC", "amadeus": "AMS.MC", "acs": "ACS.MC",
    "caixabank": "CABK.MC", "cellnex": "CLNX.MC", "ferrovial": "FER.MC",
    "grifols": "GRF.MC", "iberia": "IAG.MC", "mapfre": "MAP.MC",
    "arcelormittal": "MTS.MC", "naturgy": "NTGY.MC", "ibex": "^IBEX",
    "spain": "^IBEX", "españa": "^IBEX", "bolsa": "^IBEX",
    "euro": "EURUSD", "petróleo": "CL=F", "petroleo": "CL=F",
}


def fetch_headlines(days_back=1):
    """Descarga headlines de fuentes RSS."""
    headlines = []
    cutoff = datetime.now() - timedelta(days=days_back)

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = f"{title}. {summary}"[:512]  # Truncar para FinBERT
                pub_date = datetime.now()
                try:
                    import time
                    pub_date = datetime(*entry.published_parsed[:6])
                except Exception:
                    pass
                if pub_date >= cutoff:
                    headlines.append({
                        "title": title,
                        "text": text,
                        "date": pub_date.strftime("%Y-%m-%d"),
                        "source": feed.feed.get("title", url),
                    })
        except Exception as e:
            print(f"Error RSS {url}: {e}")

    print(f"Headlines descargados: {len(headlines)}")
    return headlines


def apply_finbert(headlines, batch_size=16):
    """Aplica FinBERT para scoring de sentimiento."""
    try:
        from transformers import pipeline
        finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=-1,  # CPU
        )
        print("FinBERT cargado correctamente.")
    except Exception as e:
        print(f"FinBERT no disponible: {e}. Usando heurísticas.")
        return apply_heuristic_sentiment(headlines)

    scored = []
    texts = [h["text"] for h in headlines]

    # Batch inference
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            preds = finbert(batch, truncation=True, max_length=512)
            results.extend(preds)
        except Exception:
            results.extend([{"label": "neutral", "score": 0.5}] * len(batch))

    sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
    for h, pred in zip(headlines, results):
        label = pred.get("label", "neutral").lower()
        score = pred.get("score", 0.5)
        sentiment_num = sentiment_map.get(label, 0)
        weighted_score = sentiment_num * score
        scored.append({
            **h,
            "sentiment_label": label,
            "sentiment_score": weighted_score,
            "confidence": score,
        })

    return scored


def apply_heuristic_sentiment(headlines):
    """Sentimiento heurístico cuando FinBERT no está disponible."""
    positive_words = ["sube", "gana", "beneficio", "crecimiento", "récord", "máximo",
                      "profit", "gain", "growth", "record", "rise", "rally", "bull",
                      "dividend", "expansion", "contrato", "acuerdo", "mejora"]
    negative_words = ["baja", "cae", "pérdida", "quiebra", "mínimo", "crisis", "riesgo",
                      "loss", "fall", "decline", "bear", "crash", "debt", "cut",
                      "reducción", "despido", "sanción", "multa", "impago"]

    scored = []
    for h in headlines:
        text_lower = h["text"].lower()
        pos = sum(w in text_lower for w in positive_words)
        neg = sum(w in text_lower for w in negative_words)
        if pos > neg:
            sentiment = "positive"
            score = min(0.5 + (pos - neg) * 0.1, 0.95)
        elif neg > pos:
            sentiment = "negative"
            score = min(0.5 + (neg - pos) * 0.1, 0.95)
        else:
            sentiment = "neutral"
            score = 0.5

        sentiment_map = {"positive": 1, "negative": -1, "neutral": 0}
        scored.append({
            **h,
            "sentiment_label": sentiment,
            "sentiment_score": sentiment_map[sentiment] * score,
            "confidence": score,
        })
    return scored


def assign_tickers(scored_headlines):
    """Asigna tickers a cada headline basado en menciones de entidades."""
    assigned = []
    for h in scored_headlines:
        text_lower = h["text"].lower()
        mentioned_tickers = set()
        for entity, ticker in ENTITY_MAP.items():
            if entity in text_lower:
                mentioned_tickers.add(ticker)
        if not mentioned_tickers:
            mentioned_tickers.add("^IBEX")  # Fallback al índice

        for ticker in mentioned_tickers:
            assigned.append({
                **h,
                "ticker": ticker,
            })
    return assigned


def aggregate_daily_sentiment(assigned_headlines):
    """Agrega sentimiento por ticker y día."""
    if not assigned_headlines:
        return pd.DataFrame()

    df = pd.DataFrame(assigned_headlines)
    agg = df.groupby(["date", "ticker"]).agg(
        sentiment_daily=("sentiment_score", "mean"),
        n_articles=("sentiment_score", "count"),
        sentiment_max=("sentiment_score", "max"),
        sentiment_min=("sentiment_score", "min"),
    ).reset_index()

    # Rolling sentiment (7d y 30d)
    agg = agg.sort_values(["ticker", "date"])
    for ticker_group, grp in agg.groupby("ticker"):
        idx = grp.index
        agg.loc[idx, "sentiment_7d"] = grp["sentiment_daily"].rolling(7, min_periods=1).mean().values
        agg.loc[idx, "sentiment_30d"] = grp["sentiment_daily"].rolling(30, min_periods=1).mean().values
        agg.loc[idx, "sentiment_momentum"] = (
            grp["sentiment_daily"].rolling(7, min_periods=1).mean().values -
            grp["sentiment_daily"].rolling(30, min_periods=1).mean().values
        )

    out_path = DATA_DIR / "sentiment_features.parquet"
    agg.to_parquet(out_path)
    print(f"✓ Sentimiento guardado: {len(agg)} registros")
    return agg


def generate_synthetic_sentiment(tickers, n_days=1500, seed=42):
    """Genera sentimiento sintético para entrenamiento cuando no hay datos reales."""
    print("Generando sentimiento sintético...")
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2019-01-02", periods=n_days, freq="B")
    records = []
    for ticker in tickers:
        # Tendencia con ruido
        trend = np.cumsum(rng.normal(0, 0.01, n_days))
        daily = np.tanh(trend + rng.normal(0, 0.3, n_days))
        for i, date in enumerate(dates):
            records.append({
                "date": str(date.date()),
                "ticker": ticker,
                "sentiment_daily": round(float(daily[i]), 4),
                "n_articles": int(rng.poisson(5)),
                "sentiment_7d": round(float(np.mean(daily[max(0,i-7):i+1])), 4),
                "sentiment_30d": round(float(np.mean(daily[max(0,i-30):i+1])), 4),
                "sentiment_momentum": round(float(
                    np.mean(daily[max(0,i-7):i+1]) - np.mean(daily[max(0,i-30):i+1])), 4),
            })
    df = pd.DataFrame(records)
    df.to_parquet(DATA_DIR / "sentiment_features.parquet")
    print(f"✓ Sentimiento sintético: {len(df):,} registros")
    return df


if __name__ == "__main__":
    import os
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("=== AlphaSignal — Sentiment Pipeline ===")
    tickers = ["^IBEX", "SAN.MC", "BBVA.MC", "ITX.MC", "TEF.MC", "IBE.MC",
               "REP.MC", "AMS.MC", "ACS.MC", "CABK.MC", "CLNX.MC", "GRF.MC"]
    # Try RSS headlines with heuristics (no FinBERT download needed for portfolio demo)
    headlines = fetch_headlines(days_back=30)
    if headlines:
        scored = apply_heuristic_sentiment(headlines)
        assigned = assign_tickers(scored)
        aggregate_daily_sentiment(assigned)
        print(f"✓ Procesadas {len(headlines)} headlines con heurísticas")
    else:
        print("Generando sentimiento sintético para entrenamiento...")
        generate_synthetic_sentiment(tickers)
