"""Dynamic model registry and loader for any local PyTorch / HuggingFace transformer architecture."""

import os
import gc
import re
import glob
import torch
import urllib.request
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, Any, Optional, Tuple, List
from app.steering.hooks import ActivationHookManager


SUPPORTED_PRESETS = {
    "phi-3-mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "display_name": "Phi-3 Mini (3.8B - Microsoft)",
        "architecture": "Phi3ForCausalLM",
        "default_layers": [10, 12, 14, 16],
        "hidden_size": 3072,
        "layer_count": 32
    },
    "gpt2": {
        "model_id": "gpt2",
        "display_name": "GPT-2 Base (124M - OpenAI)",
        "architecture": "GPT2LMHeadModel",
        "default_layers": [4, 6, 8, 10],
        "hidden_size": 768,
        "layer_count": 12
    }
}


class ModelLoader:
    _active_model: Optional[Any] = None
    _active_tokenizer: Optional[Any] = None
    _active_hook_manager: Optional[ActivationHookManager] = None
    _active_model_key: Optional[str] = None
    _active_metadata: Dict[str, Any] = {}

    @classmethod
    def get_hf_hub_cache_dir(cls) -> str:
        """Get the HuggingFace cache directory for models."""
        env_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        if env_dir and os.path.exists(env_dir):
            return env_dir
        home = os.path.expanduser("~")
        return os.path.join(home, ".cache", "huggingface", "hub")

    @classmethod
    def scan_hf_cache(cls) -> List[Dict[str, Any]]:
        """Scan HuggingFace cache directory for locally available models."""
        cache_dir = cls.get_hf_hub_cache_dir()
        discovered = []

        if not os.path.exists(cache_dir):
            return discovered

        for item in os.listdir(cache_dir):
            if not item.startswith("models--"):
                continue
            item_path = os.path.join(cache_dir, item)
            if not os.path.isdir(item_path):
                continue

            # Parse repo id: e.g. models--microsoft--Phi-3-mini-4k-instruct -> microsoft/Phi-3-mini-4k-instruct
            parts = item[len("models--"):].split("--")
            repo_id = "/".join(parts) if len(parts) >= 2 else parts[0]

            # Check snapshots for model weights
            snapshots_dir = os.path.join(item_path, "snapshots")
            has_weights = False
            has_tokenizer = False
            weight_files = []
            total_size_mb = 0.0

            if os.path.exists(snapshots_dir):
                for snap in os.listdir(snapshots_dir):
                    snap_path = os.path.join(snapshots_dir, snap)
                    if os.path.isdir(snap_path):
                        for f in os.listdir(snap_path):
                            f_lower = f.lower()
                            f_path = os.path.join(snap_path, f)
                            try:
                                size = os.path.getsize(f_path)
                                total_size_mb += size / (1024 * 1024)
                            except OSError:
                                pass

                            if f_lower.endswith(".safetensors") or f_lower.endswith(".bin") or f_lower.endswith(".pt"):
                                has_weights = True
                                weight_files.append(f)
                            if "tokenizer" in f_lower or f_lower == "vocab.json":
                                has_tokenizer = True

            # Also check if it looks like a CausalLM / text model
            is_text_model = not any(
                skip in repo_id.lower()
                for skip in ["ocr", "det", "rec", "doc", "paddle", "layoutlm"]
            )

            discovered.append({
                "repo_id": repo_id,
                "folder_name": item,
                "local_path": item_path,
                "has_weights": has_weights,
                "has_tokenizer": has_tokenizer,
                "weight_files": weight_files,
                "total_size_mb": round(total_size_mb, 1),
                "is_text_model": is_text_model,
                "is_ready": has_weights and has_tokenizer and is_text_model
            })

        return discovered

    @classmethod
    def scan_ollama_models(cls, endpoint: str = "http://127.0.0.1:11434") -> List[Dict[str, Any]]:
        """Query local Ollama instance for installed models."""
        try:
            req = urllib.request.Request(f"{endpoint}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                return [
                    {
                        "name": m.get("name"),
                        "size_gb": round(m.get("size", 0) / (1024**3), 2),
                        "modified_at": m.get("modified_at"),
                        "digest": m.get("digest", "")[:12]
                    }
                    for m in models
                ]
        except Exception:
            return []

    @classmethod
    def get_supported_catalogue(cls) -> Dict[str, Any]:
        """Return combined catalogue of presets, cached HF models, and Ollama models."""
        hf_models = cls.scan_hf_cache()
        ollama_models = cls.scan_ollama_models()

        return {
            "active_model": cls._active_metadata if cls._active_model is not None else None,
            "presets": [
                {
                    "key": key,
                    "display_name": spec["display_name"],
                    "model_id": spec["model_id"],
                    "architecture": spec["architecture"],
                    "layer_count": spec["layer_count"],
                    "hidden_size": spec["hidden_size"],
                    "default_layers": spec["default_layers"]
                }
                for key, spec in SUPPORTED_PRESETS.items()
            ],
            "discovered_local_hf": hf_models,
            "ollama_models": ollama_models
        }

    @classmethod
    def unload_current_model(cls):
        """Cleanly detach hooks and free GPU/CPU memory of previously loaded model."""
        if cls._active_hook_manager is not None:
            cls._active_hook_manager.remove_hooks()
            cls._active_hook_manager = None
        if cls._active_model is not None:
            del cls._active_model
            cls._active_model = None
        if cls._active_tokenizer is not None:
            del cls._active_tokenizer
            cls._active_tokenizer = None
        cls._active_model_key = None
        cls._active_metadata.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def load_model(
        cls,
        model_identifier: str = "phi-3-mini",
        local_files_only: bool = True,
        torch_dtype: Optional[torch.dtype] = None
    ) -> Tuple[Any, Any, ActivationHookManager]:
        """
        Load ANY local transformer model.
        model_identifier can be:
          1. A preset key (e.g. 'phi-3-mini', 'gpt2')
          2. A Hugging Face repo ID (e.g. 'microsoft/Phi-3-mini-4k-instruct', 'Qwen/Qwen2.5-0.5B-Instruct')
          3. An absolute or relative path to a local folder containing model weights & tokenizer
        """
        # Return currently active if already loaded
        if cls._active_model is not None and cls._active_model_key == model_identifier:
            return cls._active_model, cls._active_tokenizer, cls._active_hook_manager

        # Unload previous model to avoid RAM exhaustion
        cls.unload_current_model()

        # Determine target path / repo id
        target_path = model_identifier
        display_name = model_identifier
        is_local_dir = os.path.exists(model_identifier)

        if model_identifier in SUPPORTED_PRESETS:
            preset = SUPPORTED_PRESETS[model_identifier]
            target_path = preset["model_id"]
            display_name = preset["display_name"]

        print(f"[ModelLoader] Loading local model: '{target_path}' (local_files_only={local_files_only})...", flush=True)

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            target_path,
            local_files_only=local_files_only
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

        # Resolve dtype: prefer bfloat16 for efficiency
        dtype = torch_dtype or torch.bfloat16

        # Load Model
        model = AutoModelForCausalLM.from_pretrained(
            target_path,
            dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
            local_files_only=local_files_only
        )
        model.eval()

        # Attach dynamic activation hook manager
        hook_manager = ActivationHookManager(model)
        num_layers = hook_manager.num_layers

        # Dynamically inspect config for metadata
        config = getattr(model, "config", None)
        hidden_size = getattr(
            config, "hidden_size",
            getattr(config, "n_embd", getattr(config, "d_model", 3072 if num_layers > 20 else 768))
        )
        arch = model.__class__.__name__

        # Compute recommended target layers (middle 35% - 65% of transformer depth)
        recommended_layers = [
            round(num_layers * p)
            for p in [0.35, 0.42, 0.50, 0.58, 0.65]
        ]
        # Deduplicate & bound
        recommended_layers = sorted(list(set(
            min(max(0, l), num_layers - 1) for l in recommended_layers
        )))

        metadata = {
            "model_key": model_identifier,
            "target_path": target_path,
            "display_name": display_name,
            "architecture": arch,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "layer_module_path": hook_manager.layer_module_path,
            "recommended_layers": recommended_layers,
            "dtype": str(dtype),
            "device": "cpu"
        }

        cls._active_model = model
        cls._active_tokenizer = tokenizer
        cls._active_hook_manager = hook_manager
        cls._active_model_key = model_identifier
        cls._active_metadata = metadata

        print(f"[ModelLoader] Successfully loaded '{display_name}' ({arch}, {num_layers} layers, hidden={hidden_size}).", flush=True)
        return model, tokenizer, hook_manager
