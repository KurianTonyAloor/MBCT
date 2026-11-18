"""Calibration provider adapter for local Ollama instances."""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from app.core.types import ProviderConfig
from app.providers.base import BaseCalibrationProvider


class OllamaProvider(BaseCalibrationProvider):
    """Adapter for local Ollama server (default: http://127.0.0.1:11434)."""

    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Ollama server with a minimal non-intrusive probe."""
        start = time.time()
        endpoint = self.config.endpoint.rstrip("/")
        try:
            req = urllib.request.Request(
                f"{endpoint}/api/tags",
                headers={"User-Agent": "MBCT/1.0", **self.config.custom_headers}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                latency = round((time.time() - start) * 1000, 1)
                data = json.loads(response.read().decode())
                models = [m.get("name") for m in data.get("models", [])]
                model_exists = any(self.config.model_id in m for m in models)
                return {
                    "success": True,
                    "provider_type": "ollama",
                    "latency_ms": latency,
                    "available_models": models,
                    "target_model_available": model_exists,
                    "message": f"Connected to Ollama ({latency}ms). Model '{self.config.model_id}' available: {model_exists}"
                }
        except Exception as e:
            return {
                "success": False,
                "provider_type": "ollama",
                "error": str(e),
                "message": f"Could not reach Ollama at {endpoint}: {str(e)}"
            }

    def generate(self, prompt: str, system_prompt: str, json_mode: bool = True) -> str:
        """Call Ollama /api/generate endpoint."""
        endpoint = self.config.endpoint.rstrip("/")
        url = f"{endpoint}/api/generate"
        
        payload: Dict[str, Any] = {
            "model": self.config.model_id,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
        if json_mode:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "MBCT/1.0",
            **self.config.custom_headers
        }
        req = urllib.request.Request(url, data=data, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                result = json.loads(response.read().decode())
                return result.get("response", "")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            raise RuntimeError(f"Ollama HTTP {e.code}: {err_body}")
        except Exception as e:
            raise RuntimeError(f"Error querying Ollama provider: {e}")
