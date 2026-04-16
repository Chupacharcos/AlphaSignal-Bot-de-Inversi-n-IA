# AlphaSignal — Bot de Inversión IA

Sistema **three-source ensemble** para el IBEX35: análisis técnico (LightGBM, 25+ indicadores), sentimiento de noticias financieras y eventos geopolíticos en tiempo real (**GDELT**). Genera señales **COMPRA/MANTIENE/VENDE** con explicabilidad SHAP y actualización diaria automática.

## Demo en vivo

[adrianmoreno-dev.com/demo/alphasignal](https://adrianmoreno-dev.com/demo/alphasignal)

## Arquitectura

```
yfinance (IBEX35 + 15 tickers, 5 años OHLCV)
    └── 25+ indicadores técnicos (RSI, MACD, Bollinger, ATR, Stochastic, OBV, EMA)
            ├── GDELT API v2 → puntuación de tensión geopolítica por país/empresa
            ├── Heuristic sentiment (keywords financieros en titulares RSS)
            └── LightGBM (Optuna 35 trials)
                    └── Ensemble three-source → COMPRA / MANTIENE / VENDE + SHAP
```

## Stack

| Componente | Tecnología |
|---|---|
| Datos de mercado | yfinance (IBEX35 + 15 componentes, 5 años) |
| Análisis técnico | 25+ indicadores: RSI, MACD, Bollinger, ATR, Stoch, OBV, EMA 20/50/200 |
| Geopolítica | GDELT Project API v2 (público, sin API key) |
| Sentimiento | Heuristic keyword-based (RSS titulares financieros) |
| Modelo | LightGBM, Optuna 35 trials, TimeSeriesSplit |
| Explicabilidad | SHAP TreeExplainer por ticker |
| API | FastAPI (puerto 8005) |
| Actualización | Cron diario 19:00 UTC (lunes-viernes) |

## Tickers IBEX35

```
^IBEX  SAN.MC  BBVA.MC  ITX.MC  IBE.MC  REP.MC  AMS.MC  
TEF.MC  CABK.MC  ANA.MC  GRF.MC  FER.MC  ACS.MC  MAP.MC  ELE.MC
```

## Indicadores técnicos

| Grupo | Indicadores |
|---|---|
| Momentum | RSI(14), Stochastic(%K/%D), MACD(12,26,9), Rate of Change |
| Tendencia | EMA20, EMA50, EMA200, ADX |
| Volatilidad | Bollinger Bands (20,2), ATR(14), Volatilidad 20d |
| Volumen | OBV, Volume ratio |
| Precio | Returns 1d/5d/21d, High/Low ratio |

## Métricas

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.5280 |
| Sharpe estrategia | 2.558 |
| Sharpe Buy&Hold | 1.827 |
| Tickers | 15 (IBEX35) |
| Datos | 5 años OHLCV |
| Update | Diario 21:00h España |

> El AUC ~0.53 es esperado para predicción de dirección de mercado. El valor del sistema está en el Sharpe 2.56 vs 1.83 del Buy&Hold, reflejando una mejor gestión del riesgo.

## Nota sobre FinBERT

El diseño original incluía ProsusAI/finbert (441MB). Para el demo de portfolio se usa análisis heurístico basado en keywords financieros (positivo/negativo) sobre titulares RSS. En producción se integraría FinBERT con inferencia por lotes nocturna.

## Endpoints

```
GET /ml/signals/health              Estado del servicio
GET /ml/signals/today               15 señales IBEX35 con SHAP
GET /ml/signals/signal/{ticker}     Señal detallada por ticker
GET /ml/signals/geopolitics/now     Eventos GDELT actuales
GET /ml/signals/sentiment/{ticker}  Sentimiento por ticker
GET /ml/signals/backtest/results    Backtesting histórico
GET /ml/signals/stats               Métricas del modelo
```

### Ejemplo response

```json
{
  "ticker": "SAN.MC",
  "signal": "COMPRA",
  "prob_up": 0.61,
  "confidence": 0.22,
  "price": {
    "close": 4.52,
    "ret_1d": 0.88,
    "rsi_14": 58.3,
    "macd_hist": 0.021
  },
  "sentiment": { "sentiment_7d": 0.52, "tension_score": 0.1 }
}
```

## Instalación

```bash
git clone https://github.com/Chupacharcos/AlphaSignal-Bot-de-Inversi-n-IA.git
cd AlphaSignal-Bot-de-Inversi-n-IA
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Descargar datos y entrenar
python scripts/download_market_data.py    # yfinance 5 años IBEX35
python scripts/compute_indicators.py      # 25+ indicadores técnicos
python scripts/process_news_sentiment.py  # Sentimiento heurístico
python scripts/train_all.py               # LightGBM + Optuna → artifacts/

# Arrancar API
uvicorn api:app --port 8005

# Cron diario (crontab)
# 0 19 * * 1-5 cd /var/www/alphasignal && python scripts/download_market_data.py && ...
```

## Advertencia

Las señales generadas son para fines de investigación y demostración. No constituyen asesoramiento financiero ni recomendación de inversión.

## Licencia

MIT
