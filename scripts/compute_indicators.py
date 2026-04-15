#!/usr/bin/env python3
"""
AlphaSignal — Feature Engineering
Calcula indicadores técnicos y features avanzadas para LightGBM.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def compute_technical_features(df_raw):
    """Calcula features técnicas para cada ticker."""
    print("Computando indicadores técnicos...")
    all_features = []

    for ticker, group in df_raw.groupby("ticker"):
        g = group.sort_values("date").copy()
        c = g["close"].values

        if len(c) < 30:
            continue

        # === PRECIO Y VOLUMEN ===
        g["ret_1d"] = g["close"].pct_change(1)
        g["ret_3d"] = g["close"].pct_change(3)
        g["ret_5d"] = g["close"].pct_change(5)
        g["ret_20d"] = g["close"].pct_change(20)
        g["vol_20d"] = g["ret_1d"].rolling(20).std()
        g["vol_ratio"] = g["volume"] / g["volume"].rolling(20).mean()
        g["gap_open"] = (g["open"] / g["close"].shift(1)) - 1
        g["high_low_range"] = (g["high"] - g["low"]) / g["close"]

        # === MEDIAS MÓVILES ===
        g["ema_20"] = g["close"].ewm(span=20).mean()
        g["ema_50"] = g["close"].ewm(span=50).mean()
        g["ema_200"] = g["close"].ewm(span=200).mean()
        g["price_vs_ema20"] = g["close"] / g["ema_20"] - 1
        g["price_vs_ema200"] = g["close"] / g["ema_200"] - 1
        g["ema20_vs_ema50"] = g["ema_20"] / g["ema_50"] - 1
        g["golden_cross"] = (g["ema_20"] > g["ema_50"]).astype(int)

        # === RSI(14) ===
        delta = g["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        g["rsi_14"] = 100 - (100 / (1 + rs))

        # === MACD(12,26,9) ===
        ema12 = g["close"].ewm(span=12).mean()
        ema26 = g["close"].ewm(span=26).mean()
        g["macd"] = ema12 - ema26
        g["macd_signal"] = g["macd"].ewm(span=9).mean()
        g["macd_hist"] = g["macd"] - g["macd_signal"]

        # === BOLLINGER BANDS ===
        sma20 = g["close"].rolling(20).mean()
        std20 = g["close"].rolling(20).std()
        g["bb_upper"] = sma20 + 2 * std20
        g["bb_lower"] = sma20 - 2 * std20
        g["bb_width"] = (g["bb_upper"] - g["bb_lower"]) / sma20
        g["bb_position"] = (g["close"] - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"] + 1e-10)

        # === ATR(14) ===
        tr = pd.DataFrame({
            "hl": g["high"] - g["low"],
            "hpc": abs(g["high"] - g["close"].shift(1)),
            "lpc": abs(g["low"] - g["close"].shift(1)),
        }).max(axis=1)
        g["atr_14"] = tr.rolling(14).mean()
        g["atr_pct"] = g["atr_14"] / g["close"]

        # === STOCHASTIC(14,3) ===
        low14 = g["low"].rolling(14).min()
        high14 = g["high"].rolling(14).max()
        g["stoch_k"] = 100 * (g["close"] - low14) / (high14 - low14 + 1e-10)
        g["stoch_d"] = g["stoch_k"].rolling(3).mean()

        # === OBV ===
        obv = [0]
        for i in range(1, len(g)):
            if g["close"].iloc[i] > g["close"].iloc[i-1]:
                obv.append(obv[-1] + g["volume"].iloc[i])
            elif g["close"].iloc[i] < g["close"].iloc[i-1]:
                obv.append(obv[-1] - g["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        g["obv"] = obv
        g["obv_slope"] = pd.Series(obv).diff(5).values

        # === 52-WEEK HIGH/LOW ===
        g["52w_high"] = g["close"].rolling(252).max()
        g["52w_low"] = g["close"].rolling(252).min()
        g["dist_52w_high"] = g["close"] / g["52w_high"] - 1
        g["dist_52w_low"] = g["close"] / g["52w_low"] - 1

        # === TARGETS (predecir retorno futuro) ===
        g["target_1d"] = g["ret_1d"].shift(-1)  # Retorno mañana
        g["target_3d"] = g["close"].pct_change(3).shift(-3)  # Retorno en 3 días
        g["target_7d"] = g["close"].pct_change(7).shift(-7)  # Retorno en 7 días
        g["target_direction_1d"] = (g["target_1d"] > 0).astype(int)

        all_features.append(g)

    features_df = pd.concat(all_features, ignore_index=True)
    features_df.to_parquet(DATA_DIR / "technical_features.parquet")
    print(f"✓ Features técnicas: {len(features_df):,} filas, {len(features_df.columns)} columnas")
    return features_df


def merge_macro(features_df, macro_df):
    """Fusiona features técnicas con datos macro."""
    if macro_df is None:
        return features_df

    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.reset_index()
    # Normalize index column name (could be "Date", "index", etc.)
    date_col = [c for c in macro_df.columns if c.lower() == "date"][0]
    macro_df = macro_df.rename(columns={date_col: "date"})
    macro_df["date"] = pd.to_datetime(macro_df["date"])

    features_df["date"] = pd.to_datetime(features_df["date"])
    merged = features_df.merge(macro_df, on="date", how="left")

    # Lags para macro
    for col in ["WTI", "EURUSD", "VIX", "TNX"]:
        if col in merged.columns:
            merged[f"{col}_ret_1d"] = merged[col].pct_change(1)
            merged[f"{col}_ret_5d"] = merged[col].pct_change(5)

    merged.to_parquet(DATA_DIR / "technical_features.parquet")
    print(f"✓ Merged con macro: {len(merged):,} filas")
    return merged


if __name__ == "__main__":
    print("=== AlphaSignal — Feature Engineering ===")
    if not (DATA_DIR / "market_raw.parquet").exists():
        print("Descargando datos primero...")
        import subprocess, sys
        subprocess.run([sys.executable, str(Path(__file__).parent / "download_market_data.py")])

    df_raw = pd.read_parquet(DATA_DIR / "market_raw.parquet")
    print(f"Datos cargados: {len(df_raw):,} registros, {df_raw['ticker'].nunique()} tickers")

    macro_df = None
    if (DATA_DIR / "macro_raw.parquet").exists():
        macro_df = pd.read_parquet(DATA_DIR / "macro_raw.parquet")

    features = compute_technical_features(df_raw)
    features = merge_macro(features, macro_df)
    print("\n✓ Feature engineering completado.")
