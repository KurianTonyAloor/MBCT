"""Calibration Runner that orchestrates dataset generation, validation, vector derivation and MCP generation across any local model."""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional
from app.core.types import (
    ModelCalibrationProfile,
    SteeringVectorArtifact,
    QualityPreset,
    JobState
)
from app.providers.ollama_provider import OllamaProvider
from app.dataset.generator import DatasetGenerator
from app.models.registry import ModelLoader
from app.steering.deriver import VectorDeriver


class CalibrationService:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.profiles_dir = os.path.join(artifacts_dir, "profiles")
        self.datasets_dir = os.path.join(artifacts_dir, "datasets")
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)

        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.current_mcp: Optional[ModelCalibrationProfile] = None

        # Load default MCP for phi3 if present
        self.load_mcp_for_model("phi-3-mini")

    @staticmethod
    def clean_model_key(model_key: str) -> str:
        """Sanitize any model key, HF repo id, or file path into a clean identifier."""
        # Strip path or url prefixes if any
        base = os.path.basename(model_key.rstrip("/\\"))
        clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_key)
        # Avoid ultra-long filenames
        if len(clean) > 50:
            clean = clean[-50:]
        return clean.strip("_").lower()

    def load_mcp_for_model(self, model_key: str) -> Optional[ModelCalibrationProfile]:
        """Attempt to find and load a persisted MCP matching the given model."""
        clean = self.clean_model_key(model_key)
        candidates = [
            os.path.join(self.profiles_dir, f"mcp_{clean}.json"),
            os.path.join(self.profiles_dir, f"mcp_{model_key}.json"),
        ]
        # Legacy fallback for phi-3
        if "phi" in clean or "phi3" in clean:
            candidates.append(os.path.join(self.profiles_dir, "mcp_phi3.json"))

        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        mcp = ModelCalibrationProfile(**data)
                        self.current_mcp = mcp
                        print(f"[CalibrationService] Loaded MCP '{mcp.mcp_id}' for model '{model_key}' from {path}.", flush=True)
                        return mcp
                except Exception as e:
                    print(f"[CalibrationService] Error reading {path}: {e}", flush=True)

        self.current_mcp = None
        print(f"[CalibrationService] No cached MCP found for model '{model_key}'. Ready for calibration.", flush=True)
        return None

    def calibrate_traits(
        self,
        model_key: str = "phi-3-mini",
        traits: Optional[List[str]] = None,
        target_layers: Optional[List[int]] = None,
        preset: QualityPreset = QualityPreset.BALANCED
    ) -> ModelCalibrationProfile:
        """Run full calibration across requested traits for ANY local model and persist MCP."""
        traits = traits or ["directness", "formality", "verbosity"]

        print(f"[CalibrationService] Starting calibration for traits: {traits} on model: {model_key}...", flush=True)
        model, tokenizer, hook_manager = ModelLoader.load_model(model_key)

        # If target layers not explicitly supplied, derive from model metadata
        if not target_layers:
            rec = ModelLoader._active_metadata.get("recommended_layers")
            if rec and len(rec) >= 2:
                target_layers = rec
            else:
                num_l = hook_manager.num_layers
                target_layers = [max(1, int(num_l * 0.4)), int(num_l * 0.5), min(num_l - 1, int(num_l * 0.6))]

        provider = OllamaProvider()
        generator = DatasetGenerator(provider=provider)
        deriver = VectorDeriver(model, tokenizer, hook_manager)

        all_vectors: Dict[str, Dict[int, List[float]]] = {}
        chosen_layers: Dict[str, List[int]] = {}
        strength_curves: Dict[str, Dict[str, float]] = {}
        report: Dict[str, Any] = {"traits_calibrated": {}, "model": model_key}

        for trait_id in traits:
            print(f"[CalibrationService] Calibrating trait: '{trait_id}' on layers {target_layers}...", flush=True)
            examples = generator.generate_for_trait(trait_id, quality_preset=preset)
            
            # Persist dataset
            ds_path = os.path.join(self.datasets_dir, f"calibration_{trait_id}.json")
            with open(ds_path, "w", encoding="utf-8") as f:
                json.dump([ex.model_dump() for ex in examples], f, indent=2)

            artifacts = deriver.derive_trait_vectors(trait_id, examples, target_layers=target_layers)
            
            all_vectors[trait_id] = {l: art.vector for l, art in artifacts.items()}
            chosen_layers[trait_id] = list(artifacts.keys())
            strength_curves[trait_id] = {"min_alpha": -3.0, "default_alpha": 1.5, "max_alpha": 3.0}
            
            report["traits_calibrated"][trait_id] = {
                "example_count": len(examples),
                "layers": list(artifacts.keys()),
                "layer_norms": {l: art.norm for l, art in artifacts.items()}
            }
            print(f"[CalibrationService] Trait '{trait_id}' calibrated across layers {list(artifacts.keys())}.", flush=True)

        clean = self.clean_model_key(model_key)
        mcp = ModelCalibrationProfile(
            mcp_id=f"mcp-{clean}-{int(time.time())}",
            model_id=ModelLoader._active_metadata.get("target_path", model_key),
            model_fingerprint=f"{clean}-{model.__class__.__name__}",
            traits=traits,
            chosen_layers=chosen_layers,
            vectors=all_vectors,
            strength_curves=strength_curves,
            evaluation_report=report,
            status="ready"
        )

        # Save to disk using clean model identifier
        save_path = os.path.join(self.profiles_dir, f"mcp_{clean}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(mcp.model_dump_json(indent=2))
        
        self.current_mcp = mcp
        print(f"[CalibrationService] MCP saved successfully to {save_path}.", flush=True)
        return mcp
