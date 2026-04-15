#!/usr/bin/env python3
"""
AlphaSignal — Entrenamiento LightGBM Technical + Meta-Learner
Pipeline completo: datos → features → entrenamiento → backtesting
"""
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import optuna
import shap
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              classification_report, matthews_corrcoef)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Features para cada capa
TECH_FEATURES = [
    "ret_1d", "ret_3d", "ret_5d", "ret_20d", "vol_20d", "vol_ratio",
    "gap_open", "high_low_range", "price_vs_ema20", "price_vs_ema200",
    "ema20_vs_ema50", "golden_cross", "rsi_14", "macd", "macd_hist",
    "bb_width", "bb_position", "atr_pct", "stoch_k", "stoch_d",
    "obv_slope", "dist_52w_high", "dist_52w_low",
]
NLP_FEATURES = ["sentiment_daily", "sentiment_7d", "sentiment_30d", "sentiment_momentum", "n_articles"]
GEO_FEATURES = ["tension_score", "tension_7d", "tension_30d", "tension_delta_1d", "extreme_event"]
MACRO_FEATURES = ["WTI_ret_1d", "EURUSD_ret_1d", "VIX_ret_1d", "TNX_ret_1d"]


def make_feature_cols(df):
    tech = [c for c in TECH_FEATURES if c in df.columns]
    nlp = [c for c in NLP_FEATURES if c in df.columns]
    geo = [c for c in GEO_FEATURES if c in df.columns]
    macro = [c for c in MACRO_FEATURES if c in df.columns]
    lags = [c for c in df.columns if "_lag" in c and c.split("_lag")[0] in TECH_FEATURES + NLP_FEATURES + GEO_FEATURES]
    return tech + nlp + geo + macro + lags


def train_lgbm_technical(X_train, y_train, X_val, y_val, n_trials=35):
    """Entrenamiento LightGBM técnico con TimeSeries CV."""
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 5,
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                              lgb.log_evaluation(False)])
        proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({"objective": "binary", "metric": "auc", "verbosity": -1})
    model = lgb.LGBMClassifier(**best_params)
    model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    return model, study.best_value


def backtest(df_test, model, feature_cols, ticker_col="ticker", date_col="date"):
    """
    Paper trading: simula señales y calcula métricas de trading.
    """
    results = []
    for ticker, grp in df_test.groupby(ticker_col):
        grp = grp.sort_values(date_col)
        X = grp[feature_cols].fillna(0).values
        grp = grp.copy()
        grp = grp.reset_index(drop=True)
        grp["signal_prob"] = model.predict_proba(X)[:, 1]
        grp["signal"] = (grp["signal_prob"] > 0.55).astype(int)  # threshold conservador

        # Returns de la estrategia — aplanar columnas duplicadas antes de operar
        signal_shifted = grp["signal"].shift(1).values.ravel()
        ret_col = grp["ret_1d"]
        if hasattr(ret_col, "iloc") and ret_col.ndim > 1:
            ret_col = ret_col.iloc[:, 0]  # tomar primera columna si hay duplicadas
        ret_values = ret_col.values.ravel()
        grp["strategy_ret"] = signal_shifted * ret_values
        grp["bh_ret"] = ret_values  # Buy & Hold

        results.append(grp)

    all_results = pd.concat(results, ignore_index=True)

    # Métricas agregadas
    strat_daily = all_results.groupby(date_col)["strategy_ret"].mean()
    bh_daily = all_results.groupby(date_col)["bh_ret"].mean()

    def sharpe(rets, periods=252):
        r = rets.dropna()
        if len(r) < 10:
            return 0.0
        return float(r.mean() / (r.std() + 1e-10) * np.sqrt(periods))

    def max_drawdown(rets):
        cum = (1 + rets.dropna()).cumprod()
        roll_max = cum.cummax()
        dd = (cum - roll_max) / roll_max
        return float(dd.min())

    strat_sharpe = sharpe(strat_daily)
    bh_sharpe = sharpe(bh_daily)
    strat_dd = max_drawdown(strat_daily)
    strat_total = float((1 + strat_daily.dropna()).prod() - 1)
    bh_total = float((1 + bh_daily.dropna()).prod() - 1)

    hit_rate = float((all_results["strategy_ret"] > 0).mean())

    return {
        "strategy": {
            "total_return_pct": round(strat_total * 100, 2),
            "sharpe_ratio": round(strat_sharpe, 3),
            "max_drawdown_pct": round(strat_dd * 100, 2),
            "hit_rate_pct": round(hit_rate * 100, 2),
        },
        "buy_hold": {
            "total_return_pct": round(bh_total * 100, 2),
            "sharpe_ratio": round(bh_sharpe, 3),
        },
        "alpha": round((strat_total - bh_total) * 100, 2),
        "n_trades": int((all_results["signal"] == 1).sum()),
        "n_dates": int(all_results[date_col].nunique()),
    }


def main():
    print("=== AlphaSignal — Entrenamiento ===")

    # Verificar datos
    final_path = DATA_DIR / "final_dataset.parquet"
    if not final_path.exists():
        print("Ejecutando pipeline de datos...")
        import subprocess, sys
        scripts_dir = Path(__file__).parent
        for script in ["download_market_data.py", "compute_indicators.py",
                        "process_news_sentiment.py", "download_gdelt.py",
                        "merge_all_features.py"]:
            print(f"  Ejecutando {script}...")
            subprocess.run([sys.executable, str(scripts_dir / script)])

    df = pd.read_parquet(final_path)
    print(f"Dataset: {len(df):,} filas, {df['ticker'].nunique()} tickers")

    feature_cols = make_feature_cols(df)
    target_col = "target_direction_1d"

    # Evitar duplicar ret_1d (ya está en feature_cols)
    extra_cols = [c for c in [target_col, "ticker", "date", "ret_1d"] if c not in feature_cols]
    df_model = df[feature_cols + extra_cols].dropna(subset=[target_col])
    df_model = df_model.sort_values(["ticker", "date"])
    print(f"Tras dropna target: {len(df_model):,} filas")

    # Split temporal: 70% train, 15% val, 15% test
    dates_sorted = sorted(df_model["date"].unique())
    n = len(dates_sorted)
    t1 = dates_sorted[int(n * 0.70)]
    t2 = dates_sorted[int(n * 0.85)]

    train = df_model[df_model["date"] < t1]
    val = df_model[(df_model["date"] >= t1) & (df_model["date"] < t2)]
    test = df_model[df_model["date"] >= t2]

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    X_train = train[feature_cols].fillna(0).values
    y_train = train[target_col].values
    X_val = val[feature_cols].fillna(0).values
    y_val = val[target_col].values
    X_test = test[feature_cols].fillna(0).values
    y_test = test[target_col].values

    print(f"Target balance: {y_train.mean():.2%} subida")

    # Entrenamiento
    print("\nEntrenando LightGBM (Optuna 35 trials)...")
    model, best_auc = train_lgbm_technical(X_train, y_train, X_val, y_val, n_trials=35)
    print(f"  Mejor AUC-Val: {best_auc:.4f}")

    # Evaluación test
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba > 0.5).astype(int)
    auc = roc_auc_score(y_test, test_proba)
    acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    mcc = matthews_corrcoef(y_test, test_pred)
    print(f"\n=== Métricas Test ===")
    print(f"  AUC: {auc:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")

    # Backtesting
    print("\nEjecutando backtesting...")
    bt = backtest(test.copy(), model, feature_cols)
    print(f"  Strategy: {bt['strategy']['total_return_pct']}% | Sharpe: {bt['strategy']['sharpe_ratio']}")
    print(f"  Buy&Hold:  {bt['buy_hold']['total_return_pct']}% | Sharpe: {bt['buy_hold']['sharpe_ratio']}")
    print(f"  Alpha: {bt['alpha']}% | Hit rate: {bt['strategy']['hit_rate_pct']}%")

    # Guardar modelo
    joblib.dump(model, MODELS_DIR / "lgbm_technical.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

    # SHAP importances
    print("\nCalculando SHAP...")
    sample = X_test[:min(500, len(X_test))]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    fi = sorted(zip(feature_cols, mean_abs), key=lambda x: -x[1])

    # Guardar métricas
    metrics = {
        "test_auc": float(auc),
        "test_accuracy": float(acc),
        "test_f1": float(f1),
        "test_mcc": float(mcc),
        "val_auc": float(best_auc),
        "backtesting": bt,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "tickers": df_model["ticker"].unique().tolist(),
        "feature_cols": feature_cols,
        "feature_importance": {k: float(v) for k, v in fi[:20]},
        "top_features": [k for k, v in fi[:10]],
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generar señales actuales (últimas filas por ticker)
    latest = df_model.sort_values(["ticker", "date"]).groupby("ticker").tail(1)
    X_latest = latest[feature_cols].fillna(0).values
    latest_proba = model.predict_proba(X_latest)[:, 1]
    latest = latest.copy()
    latest["signal_prob"] = latest_proba

    signals = []
    for (_, row), prob in zip(latest.iterrows(), latest_proba):
        if prob > 0.60:
            signal = "COMPRA"
        elif prob < 0.40:
            signal = "VENDE"
        else:
            signal = "MANTIENE"
        signals.append({
            "ticker": row["ticker"],
            "date": str(row["date"]),
            "signal": signal,
            "prob_up": round(float(prob), 4),
            "confidence": round(abs(float(prob) - 0.5) * 2, 3),
        })

    with open(MODELS_DIR / "signals_cache.json", "w") as f:
        json.dump({"signals": signals, "updated": str(pd.Timestamp.now())}, f, indent=2)

    print(f"\n✓ Entrenamiento completado. AUC: {auc:.4f}")
    print(f"✓ {len(signals)} señales generadas")


if __name__ == "__main__":
    main()
