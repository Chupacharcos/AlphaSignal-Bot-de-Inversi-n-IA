#!/usr/bin/env python3
"""AlphaSignal — Pipeline inicial completo (ejecutar una sola vez)."""
import subprocess, sys
from pathlib import Path

scripts = Path(__file__).parent
steps = [
    ("Descargando datos de mercado...", "download_market_data.py"),
    ("Calculando indicadores técnicos...", "compute_indicators.py"),
    ("Procesando sentimiento de noticias...", "process_news_sentiment.py"),
    ("Descargando eventos GDELT...", "download_gdelt.py"),
    ("Fusionando features...", "merge_all_features.py"),
    ("Entrenando modelos...", "train_all.py"),
]
for msg, script in steps:
    print(f"\n{'='*50}")
    print(f"  {msg}")
    print(f"{'='*50}")
    result = subprocess.run([sys.executable, str(scripts / script)])
    if result.returncode != 0:
        print(f"ERROR en {script}")
        break
print("\n✓ Pipeline inicial completado.")
