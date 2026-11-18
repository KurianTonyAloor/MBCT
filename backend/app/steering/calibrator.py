"""Calibration Runner that orchestrates dataset generation, validation, vector derivation and MCP generation."""

import os
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

        # Load any existing MCP
        self._load_cached_mcp()

    def _load_cached_mcp(self):
        mcp_path = os.path.join(self.profiles_dir, "mcp_phi3.json")
        if os.path.exists(mcp_path):
            try:
                with open(mcp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.current_mcp = ModelCalibrationProfile(**data)
                    print(f"Loaded existing MCP '{self.current_mcp.mcp_id}' from disk.", flush=True)
            except Exception as e:
                print(f"Failed to load cached MCP: {e}", flush=True)

    def calibrate_traits(
        self,
        model_key: str = "phi-3-mini",
        traits: Optional[List[str]] = None,
        target_layers: Optional[List[int]] = None,
        preset: QualityPreset = QualityPreset.BALANCED
    ) -> ModelCalibrationProfile:
        """Run full calibration across requested traits and persist MCP."""
        traits = traits or ["directness", "formality", "verbosity"]
        target_layers = target_layers or [10, 12, 14, 16]

        print(f"Starting calibration for traits: {traits} on model: {model_key}...", flush=True)
        model, tokenizer, hook_manager = ModelLoader.load_model(model_key)
        provider = OllamaProvider()
        generator = DatasetGenerator(provider=provider)
        deriver = VectorDeriver(model, tokenizer, hook_manager)

        all_vectors: Dict[str, Dict[int, List[float]]] = {}
        chosen_layers: Dict[str, List[int]] = {}
        strength_curves: Dict[str, Dict[str, float]] = {}
        report: Dict[str, Any] = {"traits_calibrated": {}, "model": model_key}

        for trait_id in traits:
            print(f"Calibrating trait: '{trait_id}'...", flush=True)
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
            print(f"Trait '{trait_id}' calibrated across layers {list(artifacts.keys())}.", flush=True)

        mcp = ModelCalibrationProfile(
            mcp_id=f"mcp-{model_key}-{int(time.time())}",
            model_id="microsoft/Phi-3-mini-4k-instruct",
            model_fingerprint=f"{model_key}-bfloat16-cpu",
            traits=traits,
            chosen_layers=chosen_layers,
            vectors=all_vectors,
            strength_curves=strength_curves,
            evaluation_report=report,
            status="ready"
        )

        # Save to disk
        save_path = os.path.join(self.profiles_dir, "mcp_phi3.json")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(mcp.model_dump_json(indent=2))
        
        self.current_mcp = mcp
        print(f"MCP saved successfully to {save_path}.", flush=True)
        return mcp
