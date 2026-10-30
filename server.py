"""
TenantEase AI — FastAPI Web Application & Inference Server
Serves the modern pure-white web dashboard and provides live ML prediction endpoints.

Run:
  python server.py
"""

import os
import sys
import json
import webbrowser
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Ensure project modules are importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from models.maintenance_classifier import classify_complaint
from models.llm_engine import (
    generate_maintenance_triage_llm,
    generate_retention_action_llm,
    MAINTENANCE_SYSTEM_PROMPT,
    RETENTION_SYSTEM_PROMPT,
)

app = FastAPI(
    title="TenantEase AI Platform API",
    description="Cognitive Intelligence Layer for Cynbit Technologies' TenantEase",
    version="2.4.0",
)

# Request schemas
class ComplaintRequest(BaseModel):
    text: str

class LLMTriageRequest(BaseModel):
    tenant_name: str = "Resident"
    property_name: str = "Jaipur Residency"
    category: str
    urgency_score: int
    is_emergency: bool
    complaint_text: str
    api_key: Optional[str] = None
    provider: str = "gemini"

class LLMRetentionRequest(BaseModel):
    tenant_name: str
    property_name: str
    monthly_rent: int
    days_left: int
    open_complaints: int
    delay_trend: float
    action_type: str = "whatsapp"
    api_key: Optional[str] = None
    provider: str = "gemini"

# Endpoints
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "TenantEase AI Engine", "version": "2.4.0"}

@app.post("/api/triage-complaint")
def api_triage_complaint(req: ComplaintRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    result = classify_complaint(req.text)
    return result

@app.post("/api/llm/triage")
def api_llm_triage(req: LLMTriageRequest):
    res = generate_maintenance_triage_llm(
        tenant_name=req.tenant_name,
        property_name=req.property_name,
        category=req.category,
        urgency_score=req.urgency_score,
        is_emergency=req.is_emergency,
        complaint_text=req.complaint_text,
        api_key=req.api_key,
        provider=req.provider,
    )
    return res

@app.post("/api/llm/retention")
def api_llm_retention(req: LLMRetentionRequest):
    res = generate_retention_action_llm(
        tenant_name=req.tenant_name,
        property_name=req.property_name,
        monthly_rent=req.monthly_rent,
        days_left=req.days_left,
        open_complaints=req.open_complaints,
        delay_trend=req.delay_trend,
        action_type=req.action_type,
        api_key=req.api_key,
        provider=req.provider,
    )
    return res

@app.get("/api/tenants")
def get_tenants():
    json_path = os.path.join(BASE_DIR, "data", "tenants_enriched.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Dataset snapshot not found")

# Serve data directory for static JSONs
app.mount("/data", StaticFiles(directory=os.path.join(BASE_DIR, "data")), name="data")

# Serve static frontend
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


def start_server():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    url = f"http://localhost:{port}" if host in ["0.0.0.0", "127.0.0.1"] else f"http://{host}:{port}"
    print("=" * 65)
    print(" TENANTEASE AI -- HIGH-PRECISION WEB COMMAND CENTER")
    print(f" Starting server on: {host}:{port}")
    print("=" * 65)
    
    # Automatically open browser if running locally on desktop
    if sys.platform == "win32" and "PORT" not in os.environ:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
