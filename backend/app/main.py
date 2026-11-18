"""FastAPI application for the Model-Independent Behaviour Steering System (MBCT)."""

import os
import sys
import asyncio
import uuid
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.core.types import (
    UniversalBehaviourProfile,
    ModelCalibrationProfile,
    CompiledSteeringPlan,
    QualityPreset
)
from app.prompt_registry.registry import list_traits
from app.models.registry import ModelLoader
from app.providers.ollama_provider import OllamaProvider
from app.steering.calibrator import CalibrationService
from app.profiles.compiler import PersonalityCompiler
from app.steering.controller import SteeringController


app = FastAPI(
    title="MBCT: Model-Independent Activation Steering Engine",
    description="Inference-time personality layer via contrastive residual-stream activation steering.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "MBCT API online. UI index.html not found."}


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
calibration_service = CalibrationService(artifacts_dir="artifacts")
current_ubp = UniversalBehaviourProfile(
    profile_id="default-assistant",
    traits={
        "directness": 0.85,
        "formality": 0.35,
        "verbosity": 0.40,
        "empathy": 0.60,
        "initiative": 0.50
    }
)
current_plan: Optional[CompiledSteeringPlan] = None
steering_controller: Optional[SteeringController] = None
active_model_key = "phi-3-mini"
master_steering_enabled = True

# Async calibration job registry  {job_id: {"status": str, "progress": str, "result": any}}
calibration_jobs: Dict[str, Dict] = {}

# Personality presets
PRESETS = {
    "direct_architect": {
        "label": "Direct Architect",
        "icon": "🏗️",
        "description": "High directness & formality, low verbosity. No-nonsense technical authority.",
        "traits": {"directness": 0.95, "formality": 0.80, "verbosity": 0.30, "empathy": 0.40, "initiative": 0.70}
    },
    "friendly_mentor": {
        "label": "Friendly Mentor",
        "icon": "🎓",
        "description": "Warm, conversational, detailed explanations. Great for onboarding.",
        "traits": {"directness": 0.30, "formality": 0.20, "verbosity": 0.75, "empathy": 0.90, "initiative": 0.50}
    },
    "brutal_reviewer": {
        "label": "Brutal Reviewer",
        "icon": "🔍",
        "description": "Maximum directness, low empathy. Strips out all hedging.",
        "traits": {"directness": 1.00, "formality": 0.50, "verbosity": 0.25, "empathy": 0.20, "initiative": 0.80}
    },
    "neutral": {
        "label": "Neutral Baseline",
        "icon": "⚖️",
        "description": "All traits at 50% — zero steering effect. Useful as reference.",
        "traits": {"directness": 0.50, "formality": 0.50, "verbosity": 0.50, "empathy": 0.50, "initiative": 0.50}
    }
}


def get_or_init_controller() -> SteeringController:
    global steering_controller, current_plan
    if steering_controller is None:
        model, tokenizer, hook_manager = ModelLoader.load_model(active_model_key)
        steering_controller = SteeringController(model, tokenizer, hook_manager)
        steering_controller.set_master_enabled(master_steering_enabled)

        # If an MCP exists, compile and apply plan
        if calibration_service.current_mcp:
            current_plan = PersonalityCompiler.compile(current_ubp, calibration_service.current_mcp)
            steering_controller.apply_plan(current_plan, calibration_service.current_mcp)
    return steering_controller


# ---------------------------------------------------------------------------
# System Status & Models
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_system_status():
    has_mcp = calibration_service.current_mcp is not None
    m_info = ModelLoader._active_metadata
    chosen = []
    if has_mcp:
        chosen = sorted(set(
            l for layers in calibration_service.current_mcp.chosen_layers.values()
            for l in layers
        ))
    elif m_info and "recommended_layers" in m_info:
        chosen = m_info["recommended_layers"]

    return {
        "active_model": active_model_key,
        "active_model_info": m_info,
        "model_loaded": steering_controller is not None,
        "calibrated": has_mcp,
        "calibrated_traits": list(calibration_service.current_mcp.traits) if has_mcp else [],
        "master_steering_enabled": master_steering_enabled,
        "current_plan_id": current_plan.plan_id if current_plan else None,
        "chosen_layers": chosen
    }


@app.get("/api/models")
def get_models():
    """Return catalogue of presets, locally discovered HuggingFace cache models, and Ollama models."""
    catalogue = ModelLoader.get_supported_catalogue()
    catalogue["active_model_key"] = active_model_key
    catalogue["is_calibrated"] = calibration_service.current_mcp is not None
    return catalogue


class ModelLoadRequest(BaseModel):
    model_identifier: str
    local_files_only: bool = True


@app.post("/api/models/load")
async def load_model_endpoint(req: ModelLoadRequest):
    """Dynamically load any local transformer model or checkpoint."""
    global active_model_key, steering_controller, current_plan
    target = req.model_identifier.strip()
    if not target:
        raise HTTPException(status_code=400, detail="model_identifier must not be empty.")

    def _do_load():
        model, tokenizer, hook_manager = ModelLoader.load_model(
            target,
            local_files_only=req.local_files_only
        )
        return model, tokenizer, hook_manager

    try:
        model, tokenizer, hook_manager = await asyncio.to_thread(_do_load)
        active_model_key = target
        steering_controller = SteeringController(model, tokenizer, hook_manager)
        steering_controller.set_master_enabled(master_steering_enabled)

        # Check if an existing MCP matches this model
        mcp = calibration_service.load_mcp_for_model(active_model_key)
        if mcp:
            current_plan = PersonalityCompiler.compile(current_ubp, mcp)
            steering_controller.apply_plan(current_plan, mcp)
        else:
            current_plan = None

        return {
            "status": "loaded",
            "model_key": active_model_key,
            "metadata": ModelLoader._active_metadata,
            "calibrated": mcp is not None,
            "calibrated_traits": mcp.traits if mcp else [],
            "chosen_layers": list(mcp.chosen_layers.values())[0] if (mcp and mcp.chosen_layers) else ModelLoader._active_metadata.get("recommended_layers", [])
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{target}': {str(exc)}")


@app.post("/api/models/scan")
def scan_models_endpoint():
    """Rescan local HF cache and Ollama models."""
    return ModelLoader.get_supported_catalogue()


@app.get("/api/traits")
def get_traits():
    return {"traits": list_traits()}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class ProviderTestRequest(BaseModel):
    endpoint: str = "http://127.0.0.1:11434"
    model_id: str = "phi3"


@app.post("/api/providers/test")
def test_provider(req: ProviderTestRequest):
    provider = OllamaProvider()
    provider.config.endpoint = req.endpoint
    provider.config.model_id = req.model_id
    result = provider.test_connection()
    return result


# ---------------------------------------------------------------------------
# Calibration — synchronous + async job system
# ---------------------------------------------------------------------------

class CalibrationRequest(BaseModel):
    model_key: Optional[str] = None
    traits: List[str] = ["directness", "formality", "verbosity"]
    preset: QualityPreset = QualityPreset.BALANCED


@app.post("/api/calibrations/run")
async def run_calibration(req: CalibrationRequest):
    """Launch calibration as a background job. Returns job_id immediately."""
    job_id = str(uuid.uuid4())[:8]
    target_model_key = req.model_key or active_model_key
    calibration_jobs[job_id] = {
        "status": "running",
        "progress": f"Starting calibration pipeline for '{target_model_key}'...",
        "result": None,
        "error": None
    }

    async def _calibrate():
        def _do():
            try:
                calibration_jobs[job_id]["progress"] = "Generating contrastive datasets via Ollama..."
                mcp = calibration_service.calibrate_traits(
                    model_key=target_model_key,
                    traits=req.traits,
                    preset=req.preset
                )
                calibration_jobs[job_id]["progress"] = "Applying new calibration to steering controller..."
                global active_model_key, current_plan
                active_model_key = target_model_key
                ctrl = get_or_init_controller()
                current_plan = PersonalityCompiler.compile(current_ubp, mcp)
                ctrl.apply_plan(current_plan, mcp)
                calibration_jobs[job_id]["status"] = "done"
                calibration_jobs[job_id]["progress"] = "Calibration complete."
                calibration_jobs[job_id]["result"] = {
                    "mcp_id": mcp.mcp_id,
                    "traits": mcp.traits,
                    "chosen_layers": mcp.chosen_layers,
                    "report": mcp.evaluation_report
                }
            except Exception as exc:
                calibration_jobs[job_id]["status"] = "error"
                calibration_jobs[job_id]["error"] = str(exc)
                calibration_jobs[job_id]["progress"] = f"Failed: {exc}"

        await asyncio.to_thread(_do)

    asyncio.create_task(_calibrate())
    return {"job_id": job_id, "status": "running"}


@app.get("/api/calibrations/status/{job_id}")
def get_calibration_status(job_id: str):
    if job_id not in calibration_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return calibration_jobs[job_id]


@app.get("/api/calibrations/current")
def get_current_calibration():
    if not calibration_service.current_mcp:
        return {"status": "uncalibrated", "message": "No calibration profile found. Run calibration first."}
    mcp = calibration_service.current_mcp
    return {
        "status": "calibrated",
        "mcp_id": mcp.mcp_id,
        "model_id": mcp.model_id,
        "traits": mcp.traits,
        "chosen_layers": mcp.chosen_layers,
        "report": mcp.evaluation_report,
        "created_at": mcp.created_at
    }


# ---------------------------------------------------------------------------
# Behaviour Profile & Presets
# ---------------------------------------------------------------------------

@app.get("/api/profiles/behaviour")
def get_behaviour_profile():
    return current_ubp


@app.put("/api/profiles/behaviour")
def update_behaviour_profile(ubp_update: UniversalBehaviourProfile):
    global current_ubp, current_plan
    current_ubp = ubp_update
    if calibration_service.current_mcp:
        ctrl = get_or_init_controller()
        current_plan = PersonalityCompiler.compile(current_ubp, calibration_service.current_mcp)
        ctrl.apply_plan(current_plan, calibration_service.current_mcp)
    return {
        "status": "updated",
        "profile": current_ubp,
        "plan": current_plan,
        "active_weights": current_plan.resolved_layer_weights if current_plan else {}
    }


@app.get("/api/profiles/presets")
def get_presets():
    return {"presets": PRESETS}


# ---------------------------------------------------------------------------
# Generation — blocking + streaming SSE
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 60
    temperature: float = 0.0
    compare_baseline: bool = True


@app.post("/api/steering/generate")
async def generate_text(req: GenerateRequest):
    """Blocking generation — returns full response after completion."""
    ctrl = get_or_init_controller()
    formatted_prompt = f"<|user|>\n{req.prompt}<|end|>\n<|assistant|>\n"

    def _do_inference():
        steered_resp = ctrl.generate(
            formatted_prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            steered=True
        )
        baseline_resp = None
        if req.compare_baseline:
            baseline_resp = ctrl.generate(
                formatted_prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                steered=False
            )
        return steered_resp, baseline_resp

    steered_response, baseline_response = await asyncio.to_thread(_do_inference)

    return {
        "prompt": req.prompt,
        "steered_response": steered_response,
        "baseline_response": baseline_response,
        "steering_applied": ctrl.hook_manager.steering_enabled and master_steering_enabled,
        "active_weights": current_plan.resolved_layer_weights if current_plan else {},
        "compatibility_warnings": current_plan.compatibility_warnings if current_plan else []
    }


class StreamRequest(BaseModel):
    prompt: str
    max_tokens: int = 120
    temperature: float = 0.0
    steered: bool = True


@app.post("/api/steering/generate/stream")
async def generate_stream(req: StreamRequest):
    """
    Server-Sent Events streaming endpoint.
    Yields token chunks as: data: {"token": "...", "done": false}
    Final message:          data: {"token": "", "done": true, "active_weights": {...}}
    """
    ctrl = get_or_init_controller()
    formatted_prompt = f"<|user|>\n{req.prompt}<|end|>\n<|assistant|>\n"

    async def _event_generator():
        # Run the streaming generator in a thread pool to avoid blocking asyncio
        loop = asyncio.get_event_loop()
        token_queue: asyncio.Queue = asyncio.Queue()

        def _stream_tokens():
            """Runs in background thread — feeds tokens into asyncio queue."""
            try:
                for token in ctrl.generate_stream(
                    formatted_prompt,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    steered=req.steered
                ):
                    # Put token in queue (thread-safe)
                    loop.call_soon_threadsafe(token_queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)  # Sentinel

        import threading
        t = threading.Thread(target=_stream_tokens, daemon=True)
        t.start()

        while True:
            token = await token_queue.get()
            if token is None:
                break
            payload = json.dumps({"token": token, "done": False})
            yield f"data: {payload}\n\n"

        weights = current_plan.resolved_layer_weights if current_plan else {}
        done_payload = json.dumps({
            "token": "",
            "done": True,
            "steering_applied": req.steered and master_steering_enabled,
            "active_weights": weights
        })
        yield f"data: {done_payload}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ---------------------------------------------------------------------------
# Steering Toggle
# ---------------------------------------------------------------------------

class ToggleRequest(BaseModel):
    enabled: bool


@app.post("/api/steering/toggle")
def toggle_steering(req: ToggleRequest):
    global master_steering_enabled
    master_steering_enabled = req.enabled
    ctrl = get_or_init_controller()
    ctrl.set_master_enabled(req.enabled)
    return {"master_steering_enabled": master_steering_enabled}


# ---------------------------------------------------------------------------
# Dataset viewer
# ---------------------------------------------------------------------------

@app.get("/api/datasets/{trait}")
def get_dataset(trait: str):
    """Return stored contrastive dataset pairs for a given trait."""
    import glob as glob_mod
    artifacts_base = "artifacts/datasets"
    pattern = os.path.join(artifacts_base, f"calibration_{trait}.json")
    matches = glob_mod.glob(pattern)
    if not matches:
        raise HTTPException(status_code=404, detail=f"No dataset found for trait '{trait}'")
    with open(matches[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    # Dataset is stored as a top-level list of CalibrationExample objects
    if isinstance(data, list):
        pairs = data
    else:
        pairs = data.get("pairs", [])
    return {"trait": trait, "count": len(pairs), "pairs": pairs[:10]}  # Cap at 10 for UI


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8088, reload=False)
