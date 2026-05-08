"""AlphaSignal — Bot de Inversión IA — FastAPI App."""
import asyncio
import logging
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.signals import router

logger = logging.getLogger("alphasignal")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")

app = FastAPI(title="AlphaSignal — Bot de Inversión IA", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


# ── Background task: regenera señales L-V a las 19:00 UTC ────────────────────

def _seconds_until_next_market_close() -> float:
    """Próximo lunes-viernes a las 19:00 UTC."""
    now = datetime.utcnow()
    target = now.replace(hour=19, minute=0, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    while target.weekday() >= 5:  # 5=sáb, 6=dom
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def _signals_scheduler():
    """Loop autónomo: duerme hasta el próximo cierre de mercado, regenera señales."""
    logger.info("[scheduler] AlphaSignal autonomous scheduler activo (L-V 19:00 UTC)")
    while True:
        wait_s = _seconds_until_next_market_close()
        logger.info(f"[scheduler] próxima generación en {wait_s/3600:.1f}h")
        await asyncio.sleep(wait_s)
        try:
            from scripts.generate_signals import run_daily_update
            await asyncio.to_thread(run_daily_update)
            logger.info("[scheduler] señales regeneradas correctamente")
        except Exception as e:
            logger.error(f"[scheduler] error regenerando señales: {e}", exc_info=True)
            await asyncio.sleep(900)
            continue
        await asyncio.sleep(60)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_signals_scheduler())


@app.get("/health")
def health():
    return {"status": "ok", "service": "alphasignal", "port": 8005}
