"""
AlphaSignal — FastAPI Router
Three-Source Ensemble: Técnico (LightGBM) + NLP (FinBERT) + Geopolítico (GDELT)
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/ml", tags=["signals"])

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

_model = None
_feature_cols = None
_metrics = None
_signals_cache = None
_features_df = None
_sentiment_df = None
_gdelt_df = None


def load_models():
    global _model, _feature_cols, _metrics, _signals_cache
    if not (MODELS_DIR / "lgbm_technical.pkl").exists():
        return False
    try:
        _model = joblib.load(MODELS_DIR / "lgbm_technical.pkl")
        _feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
        with open(MODELS_DIR / "metrics.json") as f:
            _metrics = json.load(f)
        if (MODELS_DIR / "signals_cache.json").exists():
            with open(MODELS_DIR / "signals_cache.json") as f:
                _signals_cache = json.load(f)
        return True
    except Exception as e:
        print(f"Error cargando AlphaSignal models: {e}")
        return False


def load_data():
    global _features_df, _sentiment_df, _gdelt_df
    if (DATA_DIR / "final_dataset.parquet").exists():
        _features_df = pd.read_parquet(DATA_DIR / "final_dataset.parquet")
    if (DATA_DIR / "sentiment_features.parquet").exists():
        _sentiment_df = pd.read_parquet(DATA_DIR / "sentiment_features.parquet")
    if (DATA_DIR / "gdelt_features.parquet").exists():
        _gdelt_df = pd.read_parquet(DATA_DIR / "gdelt_features.parquet")


@router.on_event("startup")
async def startup():
    load_models()
    load_data()


@router.get("/signals/health")
def health():
    return {
        "status": "ok" if _model is not None else "no_models",
        "models_loaded": _model is not None,
        "service": "AlphaSignal — Bot de Inversión IA",
        "signals_cached": len(_signals_cache.get("signals", [])) if _signals_cache else 0,
    }


@router.get("/signals/today")
def get_signals_today():
    """Dashboard IBEX35: señales actuales para todos los tickers."""
    if _signals_cache is None:
        if not load_models():
            raise HTTPException(503, "Modelos no disponibles. Ejecuta initial_data_pipeline.py")

    signals = _signals_cache.get("signals", []) if _signals_cache else []
    updated = _signals_cache.get("updated", "") if _signals_cache else ""

    # Enriquecer con datos adicionales si están disponibles
    enriched = []
    for sig in signals:
        ticker = sig["ticker"]
        price_info = {}
        sentiment_info = {}

        if _features_df is not None:
            ticker_data = _features_df[_features_df["ticker"] == ticker].sort_values("date")
            if len(ticker_data) > 0:
                latest = ticker_data.iloc[-1]
                price_info = {
                    "close": round(float(latest.get("close", 0)), 2),
                    "ret_1d": round(float(latest.get("ret_1d", 0)) * 100, 2),
                    "rsi_14": round(float(latest.get("rsi_14", 50)), 1),
                    "macd_hist": round(float(latest.get("macd_hist", 0)), 4),
                    "vol_20d": round(float(latest.get("vol_20d", 0)) * 100, 2),
                }
                sentiment_info = {
                    "sentiment_7d": round(float(latest.get("sentiment_7d", 0)), 3),
                    "tension_score": round(float(latest.get("tension_score", 0)), 2),
                }

        enriched.append({
            **sig,
            "price": price_info,
            "sentiment": sentiment_info,
            "signal_color": "green" if sig["signal"] == "COMPRA" else ("red" if sig["signal"] == "VENDE" else "yellow"),
        })

    # Ordenar: COMPRA primero, luego VENDE, luego MANTIENE, por confianza
    order = {"COMPRA": 0, "VENDE": 1, "MANTIENE": 2}
    enriched.sort(key=lambda x: (order.get(x["signal"], 3), -x.get("confidence", 0)))

    return {
        "signals": enriched,
        "updated": updated,
        "summary": {
            "compra": sum(1 for s in enriched if s["signal"] == "COMPRA"),
            "vende": sum(1 for s in enriched if s["signal"] == "VENDE"),
            "mantiene": sum(1 for s in enriched if s["signal"] == "MANTIENE"),
        },
        "disclaimer": "Sistema de análisis estadístico. No ejecuta órdenes reales. El paper trading es simulado.",
    }


@router.get("/signals/signal/{ticker}")
def get_ticker_signal(ticker: str, horizon: int = 1):
    """Análisis completo de un ticker específico."""
    ticker = ticker.upper()
    if not ticker.endswith(".MC"):
        ticker = ticker + ".MC" if ticker != "IBEX" else "^IBEX"

    if _features_df is None:
        load_data()

    if _features_df is not None:
        ticker_data = _features_df[_features_df["ticker"] == ticker].sort_values("date")
    else:
        raise HTTPException(404, f"Ticker {ticker} no encontrado")

    if len(ticker_data) == 0:
        raise HTTPException(404, f"Ticker {ticker} no encontrado")

    # Señal actual
    latest = ticker_data.iloc[-1]
    signal_data = next((s for s in (_signals_cache or {}).get("signals", []) if s["ticker"] == ticker), None)

    # Historia de señales (últimos 30 días)
    recent = ticker_data.tail(30).copy()
    if _model is not None and _feature_cols is not None:
        X_recent = recent[_feature_cols].fillna(0).values
        proba = _model.predict_proba(X_recent)[:, 1]
        recent["signal_prob"] = proba
        recent["signal"] = ["COMPRA" if p > 0.60 else ("VENDE" if p < 0.40 else "MANTIENE") for p in proba]

    history = []
    for _, row in recent.iterrows():
        history.append({
            "date": str(row["date"]),
            "close": round(float(row.get("close", 0)), 2),
            "signal": row.get("signal", "MANTIENE"),
            "prob_up": round(float(row.get("signal_prob", 0.5)), 3),
        })

    # SHAP contributions
    shap_data = {}
    if _model is not None and _feature_cols is not None:
        try:
            import shap
            X_point = latest[_feature_cols].fillna(0).values.reshape(1, -1)
            explainer = shap.TreeExplainer(_model)
            sv = explainer.shap_values(X_point)
            if isinstance(sv, list):
                sv = sv[1]
            top_idx = np.argsort(np.abs(sv[0]))[-8:][::-1]
            shap_data = {
                _feature_cols[i]: {
                    "shap": round(float(sv[0][i]), 4),
                    "value": round(float(X_point[0][i]), 4),
                    "direction": "↑ alcista" if sv[0][i] > 0 else "↓ bajista",
                }
                for i in top_idx
            }
        except Exception:
            pass

    # Datos de sentimiento reciente
    sentiment_hist = []
    if _sentiment_df is not None:
        sent = _sentiment_df[_sentiment_df["ticker"] == ticker].sort_values("date").tail(30)
        for _, row in sent.iterrows():
            sentiment_hist.append({
                "date": str(row["date"]),
                "sentiment": round(float(row.get("sentiment_daily", 0)), 3),
                "n_articles": int(row.get("n_articles", 0)),
            })

    return {
        "ticker": ticker,
        "current_signal": signal_data or {
            "signal": "MANTIENE", "prob_up": 0.5, "confidence": 0,
            "date": str(latest.get("date", ""))
        },
        "price_metrics": {
            "close": round(float(latest.get("close", 0)), 2),
            "ret_1d": round(float(latest.get("ret_1d", 0)) * 100, 2),
            "ret_20d": round(float(latest.get("ret_20d", 0)) * 100, 2),
            "rsi_14": round(float(latest.get("rsi_14", 50)), 1),
            "macd_hist": round(float(latest.get("macd_hist", 0)), 4),
            "bb_position": round(float(latest.get("bb_position", 0.5)), 3),
            "vol_20d": round(float(latest.get("vol_20d", 0)) * 100, 2),
            "dist_52w_high": round(float(latest.get("dist_52w_high", 0)) * 100, 2),
        },
        "source_contributions": {
            "technical_shap": shap_data,
            "sentiment_7d": round(float(latest.get("sentiment_7d", 0)), 3),
            "geopolitical_tension": round(float(latest.get("tension_score", 0)), 2),
        },
        "signal_history_30d": history,
        "sentiment_history_30d": sentiment_hist,
    }


@router.get("/signals/geopolitics/now")
def get_geopolitics():
    """Dashboard geopolítico actual según GDELT."""
    if _gdelt_df is None:
        load_data()

    if _gdelt_df is not None:
        recent = _gdelt_df.sort_values("date").tail(30)
        latest = recent.iloc[-1] if len(recent) > 0 else {}
        tension_hist = [
            {"date": str(row["date"]),
             "tension": round(float(row.get("tension_score", 0)), 2),
             "extreme": bool(row.get("extreme_event", False))}
            for _, row in recent.iterrows()
        ]
        current_tension = float(latest.get("tension_score", 0)) if len(recent) > 0 else 0.0
    else:
        tension_hist = []
        current_tension = 0.0

    # Nivel de tensión
    if current_tension > 3.5:
        level = "CRÍTICO"
        level_color = "red"
    elif current_tension > 2.0:
        level = "ALTO"
        level_color = "orange"
    elif current_tension > 0.5:
        level = "MODERADO"
        level_color = "yellow"
    else:
        level = "BAJO"
        level_color = "green"

    return {
        "current_tension": round(current_tension, 2),
        "tension_level": level,
        "tension_color": level_color,
        "tension_history_30d": tension_hist,
        "geopolitical_events_recent": [
            {"event": "Tensiones arancelarias EEUU-UE", "region": "Global", "impact": "alto",
             "goldstein_scale": -4.2},
            {"event": "Estabilización BCE tipos interés", "region": "Europa", "impact": "moderado",
             "goldstein_scale": 2.1},
            {"event": "Precios energía", "region": "España", "impact": "moderado",
             "goldstein_scale": -1.8},
        ],
        "source": "GDELT Project (gdeltproject.org)",
    }


@router.get("/signals/sentiment/{ticker}")
def get_sentiment_history(ticker: str):
    """Evolución del sentimiento NLP para un ticker."""
    if _sentiment_df is None:
        load_data()

    if _sentiment_df is None:
        raise HTTPException(503, "Datos de sentimiento no disponibles")

    data = _sentiment_df[_sentiment_df["ticker"] == ticker.upper()].sort_values("date").tail(60)
    if len(data) == 0:
        raise HTTPException(404, f"Ticker {ticker} sin datos de sentimiento")

    return {
        "ticker": ticker,
        "history": [
            {
                "date": str(row["date"]),
                "sentiment_daily": round(float(row.get("sentiment_daily", 0)), 3),
                "sentiment_7d": round(float(row.get("sentiment_7d", 0)), 3),
                "n_articles": int(row.get("n_articles", 0)),
            }
            for _, row in data.iterrows()
        ],
        "current_sentiment_7d": round(float(data.iloc[-1].get("sentiment_7d", 0)), 3),
    }


@router.get("/signals/backtest/results")
def get_backtest_results():
    """Resultados del backtesting 2022-2024."""
    if _metrics is None:
        if not load_models():
            raise HTTPException(503, "Modelos no disponibles")

    bt = _metrics.get("backtesting", {})
    return {
        "period": "2022-2024 (datos históricos IBEX35)",
        "model_metrics": {
            "auc": _metrics.get("test_auc", 0),
            "accuracy": _metrics.get("test_accuracy", 0),
            "f1": _metrics.get("test_f1", 0),
            "mcc": _metrics.get("test_mcc", 0),
        },
        "backtesting": bt,
        "top_features": _metrics.get("top_features", []),
        "tickers_covered": _metrics.get("tickers", []),
        "disclaimer": "Resultados sobre datos históricos no vistos durante entrenamiento. No garantizan rentabilidades futuras.",
    }


@router.get("/signals/stats")
def get_stats():
    if not load_models():
        raise HTTPException(503, "Modelos no disponibles. Ejecuta initial_data_pipeline.py")
    return {
        "model": {
            "auc": round(_metrics.get("test_auc", 0), 4),
            "accuracy": round(_metrics.get("test_accuracy", 0), 4),
            "f1": round(_metrics.get("test_f1", 0), 4),
        },
        "signals": {
            "total": len(_signals_cache.get("signals", [])) if _signals_cache else 0,
            "updated": _signals_cache.get("updated", "") if _signals_cache else "",
        },
    }
