"""AlphaSignal — Bot de Inversión IA — FastAPI App"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.signals import router

app = FastAPI(title="AlphaSignal — Bot de Inversión IA", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok", "service": "alphasignal", "port": 8005}
