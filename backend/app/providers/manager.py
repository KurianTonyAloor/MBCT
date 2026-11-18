"""Provider manager: handles credentials, provider presets, persistence and adapter instantiation."""

import os
import json
from typing import Dict, Any, List, Optional
from app.core.types import ProviderConfig, ProviderType
from app.providers.base import BaseCalibrationProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_provider import OpenAICompatibleProvider


PROVIDER_PRESETS = {
    "ollama": {
        "display_name": "Ollama (Local Default)",
        "provider_type": ProviderType.OLLAMA,
        "endpoint": "http://127.0.0.1:11434",
        "model_id": "phi3",
        "requires_key": False,
        "description": "Runs completely locally on your machine with zero external network traffic."
    },
    "openai": {
        "display_name": "OpenAI Cloud",
        "provider_type": ProviderType.OPENAI,
        "endpoint": "https://api.openai.com/v1",
        "model_id": "gpt-4o-mini",
        "requires_key": True,
        "description": "OpenAI official chat completions API."
    },
    "deepseek": {
        "display_name": "DeepSeek API",
        "provider_type": ProviderType.OPENAI,
        "endpoint": "https://api.deepseek.com",
        "model_id": "deepseek-chat",
        "requires_key": True,
        "description": "High capability, cost-efficient contrastive data generation."
    },
    "groq": {
        "display_name": "Groq Cloud (Ultra Fast)",
        "provider_type": ProviderType.OPENAI,
        "endpoint": "https://api.groq.com/openai/v1",
        "model_id": "llama-3.3-70b-versatile",
        "requires_key": True,
        "description": "Sub-second generation for fast multi-trait calibration runs."
    },
    "openrouter": {
        "display_name": "OpenRouter",
        "provider_type": ProviderType.OPENAI,
        "endpoint": "https://openrouter.ai/api/v1",
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "requires_key": True,
        "description": "Unified gateway to hundreds of open-source and proprietary models."
    },
    "lm_studio": {
        "display_name": "LM Studio / Local vLLM",
        "provider_type": ProviderType.OPENAI,
        "endpoint": "http://127.0.0.1:1234/v1",
        "model_id": "local-model",
        "requires_key": False,
        "description": "Locally hosted OpenAI-compatible server (LM Studio, vLLM, Ollama OpenAI mode)."
    }
}


class ProviderManager:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.config_dir = os.path.join(artifacts_dir, "providers")
        os.makedirs(self.config_dir, exist_ok=True)
        self.config_file = os.path.join(self.config_dir, "active_config.json")
        self.active_config = self._load_config()

    def _load_config(self) -> ProviderConfig:
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return ProviderConfig(**data)
            except Exception as e:
                print(f"[ProviderManager] Warning: failed to load {self.config_file}: {e}")
        return ProviderConfig(
            provider_id="ollama-local",
            provider_type=ProviderType.OLLAMA,
            endpoint="http://127.0.0.1:11434",
            model_id="phi3"
        )

    def save_config(self, config: ProviderConfig) -> ProviderConfig:
        """Persist provider configuration (masking handled separately on display)."""
        self.active_config = config
        with open(self.config_file, "w", encoding="utf-8") as f:
            f.write(config.model_dump_json(indent=2))
        return self.active_config

    def get_masked_config(self) -> Dict[str, Any]:
        """Return provider config with credentials masked for UI/client consumption."""
        data = self.active_config.model_dump()
        data["api_key_masked"] = self.active_config.masked_key()
        data["has_key"] = bool(self.active_config.api_key)
        # Never expose raw key in UI summary
        data["api_key"] = self.active_config.masked_key()
        return data

    def get_provider(self, override_config: Optional[ProviderConfig] = None) -> BaseCalibrationProvider:
        """Instantiate appropriate adapter matching current or override configuration."""
        cfg = override_config or self.active_config
        
        # Check type or heuristics
        if cfg.provider_type == ProviderType.OLLAMA or ("11434" in cfg.endpoint and not cfg.api_key):
            return OllamaProvider(cfg)
        else:
            return OpenAICompatibleProvider(cfg)

    def get_presets(self) -> Dict[str, Any]:
        return PROVIDER_PRESETS
