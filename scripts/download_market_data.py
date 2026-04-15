#!/usr/bin/env python3
"""
AlphaSignal — Descarga de datos históricos de mercado
IBEX35 y sus 35 componentes principales + macro (petróleo, VIX, EUR/USD)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

IBEX35_TICKERS = [
    "^IBEX",        # IBEX35 índice
    "SAN.MC",       # Santander
    "BBVA.MC",      # BBVA
    "ITX.MC",       # Inditex
    "TEF.MC",       # Telefónica
    "IBE.MC",       # Iberdrola
    "REP.MC",       # Repsol
    "AMS.MC",       # Amadeus
    "ACS.MC",       # ACS
    "CABK.MC",      # CaixaBank
    "CLNX.MC",      # Cellnex
    "ENG.MC",       # Enagás
    "FER.MC",       # Ferrovial
    "GRF.MC",       # Grifols
    "IAG.MC",       # IAG (Iberia)
    "MAP.MC",       # Mapfre
    "MTS.MC",       # ArcelorMittal
    "NTGY.MC",      # Naturgy
    "RED.MC",       # Red Eléctrica
    "SAB.MC",       # Sabadell
    "SGRE.MC",      # Siemens Gamesa
    "SLR.MC",       # Solaria
    "UNI.MC",       # Unicaja
    "AENA.MC",      # AENA
    "BAI.MC",       # Banca March
    "ACX.MC",       # Acerinox
    "PHM.MC",       # Pharma Mar
    "VIS.MC",       # Viscofan
    "ZOT.MC",       # Zardoya Otis
    "BKIA.MC",      # Bankia (ahora parte CaixaBank)
    "MEL.MC",       # Meliá
    "ALMG.MC",      # Almirall
]

MACRO_TICKERS = {
    "WTI": "CL=F",      # Petróleo WTI
    "EURUSD": "EURUSD=X",
    "VIX": "^VIX",
    "GLD": "GLD",       # Oro (proxy)
    "TNX": "^TNX",      # Yield bono EEUU 10Y
}


def download_ibex35(start="2019-01-01"):
    """Descarga datos OHLCV de IBEX35 y componentes."""
    end = datetime.now().strftime("%Y-%m-%d")
    print(f"Descargando IBEX35 ({start} → {end})...")

    # Filtrar tickers válidos (algunos pueden no existir)
    valid_tickers = []
    for tk in IBEX35_TICKERS[:15]:  # Primeros 15 para no sobrecargar
        try:
            data = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
            if len(data) > 100:
                valid_tickers.append(tk)
                print(f"  ✓ {tk}: {len(data)} barras")
        except Exception as e:
            print(f"  ✗ {tk}: {e}")

    # Descargar ticker a ticker (robusto con MultiIndex de yfinance >= 0.2)
    print(f"\nDescargando {len(valid_tickers)} tickers...")
    dfs = []
    for tk in valid_tickers:
        try:
            df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) < 100:
                continue
            df.index = pd.to_datetime(df.index)
            # Aplanar MultiIndex (yfinance >= 0.2 siempre devuelve MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"adj close": "close"})
            df["ticker"] = tk
            df["date"] = df.index
            # Guardar solo columnas necesarias
            cols = [c for c in ["date", "ticker", "open", "high", "low", "close", "volume"] if c in df.columns]
            dfs.append(df[cols].reset_index(drop=True))
        except Exception as e:
            print(f"  \u2717 {tk}: {e}")

    if not dfs:
        print("No se pudieron descargar datos. Generando sint\u00e9ticos...")
        return generate_synthetic_market_data()

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(DATA_DIR / "market_raw.parquet")
    print(f"\u2713 {len(combined):,} registros guardados")
    return combined


def download_macro(start="2019-01-01"):
    """Descarga datos macro."""
    end = datetime.now().strftime("%Y-%m-%d")
    print("\nDescargando datos macro...")
    macro_data = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) > 100:
                close_col = df["Close"]
                # yfinance may return MultiIndex — flatten to Series
                if hasattr(close_col, "columns"):
                    close_col = close_col.iloc[:, 0]
                close_col.name = name
                macro_data[name] = close_col
                print(f"  ✓ {name} ({ticker}): {len(df)} barras")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    if macro_data:
        macro_df = pd.DataFrame(macro_data)
        macro_df.to_parquet(DATA_DIR / "macro_raw.parquet")
        print(f"✓ Macro: {len(macro_df):,} barras")
        return macro_df
    return None


def generate_synthetic_market_data(n_days=1500, n_tickers=12):
    """Genera datos OHLCV sintéticos realistas."""
    print("Generando datos de mercado sintéticos...")
    rng = np.random.RandomState(42)

    tickers_synthetic = ["^IBEX", "SAN.MC", "BBVA.MC", "ITX.MC", "TEF.MC", "IBE.MC",
                          "REP.MC", "AMS.MC", "ACS.MC", "CABK.MC", "CLNX.MC", "GRF.MC"]

    base_prices = {"^IBEX": 8500, "SAN.MC": 3.50, "BBVA.MC": 5.80, "ITX.MC": 28.0,
                   "TEF.MC": 4.20, "IBE.MC": 10.50, "REP.MC": 14.0, "AMS.MC": 60.0,
                   "ACS.MC": 27.0, "CABK.MC": 3.20, "CLNX.MC": 35.0, "GRF.MC": 8.0}

    dates = pd.date_range("2019-01-02", periods=n_days, freq="B")
    all_data = []

    # Factores macro compartidos (correlación de mercado)
    market_factor = np.cumsum(rng.normal(0.0003, 0.012, n_days))
    vol_regime = np.where(rng.uniform(0, 1, n_days) > 0.85, 2.0, 1.0)  # Volatility regime

    for ticker in tickers_synthetic[:n_tickers]:
        base = base_prices.get(ticker, 10.0)
        beta = rng.uniform(0.6, 1.4)
        idio_vol = rng.uniform(0.008, 0.02)

        returns = (beta * market_factor + np.cumsum(rng.normal(0, idio_vol, n_days))) * vol_regime
        prices = base * np.exp(returns - returns[0])

        intraday_vol = rng.uniform(0.003, 0.01, n_days) * vol_regime
        highs = prices * (1 + intraday_vol)
        lows = prices * (1 - intraday_vol)
        opens = lows + rng.uniform(0, 1, n_days) * (highs - lows)
        volumes = rng.lognormal(mean=np.log(1e6 * base / 10), sigma=0.5, size=n_days).astype(int)

        df = pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        })
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    combined.to_parquet(DATA_DIR / "market_raw.parquet")

    # También generar macro sintético
    macro = pd.DataFrame({
        "WTI": 65 + np.cumsum(rng.normal(0, 1.5, n_days)),
        "EURUSD": 1.10 + np.cumsum(rng.normal(0, 0.005, n_days)),
        "VIX": np.clip(15 + 5 * np.sin(np.linspace(0, 20, n_days)) + rng.normal(0, 3, n_days), 10, 80),
        "TNX": np.clip(1.5 + np.cumsum(rng.normal(0, 0.02, n_days)), 0.5, 5.5),
    }, index=dates)
    macro.to_parquet(DATA_DIR / "macro_raw.parquet")

    print(f"✓ Datos sintéticos: {len(combined):,} registros, {n_tickers} tickers")
    return combined


if __name__ == "__main__":
    print("=== AlphaSignal — Market Data Download ===")
    download_ibex35()
    download_macro()
    print("\n✓ Market data pipeline completado.")
