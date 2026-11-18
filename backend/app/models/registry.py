"""Model registry and loader for PyTorch transformer architectures."""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, Tuple
from app.core.types import ModelDefinition
from app.steering.hooks import ActivationHookManager


SUPPORTED_MODELS = {
    "phi-3-mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "display_name": "Phi-3 Mini (3.8B - Local PyTorch + Hooks)",
        "architecture": "Phi3ForCausalLM",
        "layer_module_path": "model.layers",
        "default_layers": [10, 12, 14, 16, 18, 20],
        "hidden_size": 3072,
        "layer_count": 32
    },
    "gpt2": {
        "model_id": "gpt2",
        "display_name": "GPT-2 (124M Base)",
        "architecture": "GPT2LMHeadModel",
        "layer_module_path": "transformer.h",
        "default_layers": [4, 6, 8, 10],
        "hidden_size": 768,
        "layer_count": 12
    }
}


class ModelLoader:
    _loaded_models: Dict[str, Tuple[Any, Any, ActivationHookManager]] = {}

    @classmethod
    def get_supported_catalogue(cls) -> list:
        return [
            {
                "key": key,
                "display_name": spec["display_name"],
                "model_id": spec["model_id"],
                "architecture": spec["architecture"],
                "layer_count": spec["layer_count"],
                "hidden_size": spec["hidden_size"],
                "activation_access": "Full residual-stream forward hook access"
            }
            for key, spec in SUPPORTED_MODELS.items()
        ]

    @classmethod
    def load_model(cls, model_key: str = "phi-3-mini"):
        if model_key in cls._loaded_models:
            return cls._loaded_models[model_key]

        spec = SUPPORTED_MODELS.get(model_key)
        if not spec:
            raise ValueError(f"Unknown model key '{model_key}'. Supported: {list(SUPPORTED_MODELS.keys())}")

        model_name = spec["model_id"]
        print(f"Loading base model: {model_name} (bfloat16, local_files_only)...", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load with bfloat16 for efficient memory footprint
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        model.eval()

        hook_manager = ActivationHookManager(model, layer_module_path=spec["layer_module_path"])
        cls._loaded_models[model_key] = (model, tokenizer, hook_manager)
        print(f"Successfully loaded {model_name} with {hook_manager.num_layers} layers.", flush=True)
        return model, tokenizer, hook_manager
