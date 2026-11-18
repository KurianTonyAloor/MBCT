"""Calibration provider adapter for OpenAI-compatible chat completion endpoints."""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from app.core.types import ProviderConfig
from app.providers.base import BaseCalibrationProvider


class OpenAICompatibleProvider(BaseCalibrationProvider):
    """
    Adapter for any OpenAI-compatible endpoint.
    Supports OpenAI, DeepSeek, Groq, OpenRouter, LM Studio, vLLM, and self-hosted runtimes.
    """

    def _resolve_url(self) -> str:
        """Ensure endpoint ends with chat/completions."""
        base = self.config.endpoint.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "MBCT/1.0"
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key.strip()}"
        headers.update(self.config.custom_headers)
        return headers

    def test_connection(self) -> Dict[str, Any]:
        """Validate connection using a minimal single-token probe to avoid token costs."""
        start = time.time()
        url = self._resolve_url()
        payload = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._get_headers())

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                latency = round((time.time() - start) * 1000, 1)
                result = json.loads(response.read().decode())
                return {
                    "success": True,
                    "provider_type": "openai_compatible",
                    "latency_ms": latency,
                    "model_id": self.config.model_id,
                    "endpoint": self.config.endpoint,
                    "message": f"Successfully verified connection to {self.config.endpoint} ({latency}ms). Model '{self.config.model_id}' ready."
                }
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            return {
                "success": False,
                "provider_type": "openai_compatible",
                "status_code": e.code,
                "error": f"HTTP {e.code}: {body[:200]}",
                "message": f"Endpoint returned error HTTP {e.code}. Check URL, API key, and model name."
            }
        except Exception as e:
            return {
                "success": False,
                "provider_type": "openai_compatible",
                "error": str(e),
                "message": f"Connection failed to {url}: {str(e)}"
            }

    def generate(self, prompt: str, system_prompt: str, json_mode: bool = True) -> str:
        """Call chat/completions endpoint."""
        url = self._resolve_url()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        payload: Dict[str, Any] = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._get_headers())

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                res = json.loads(response.read().decode())
                choices = res.get("choices", [])
                if not choices:
                    raise RuntimeError("No choices returned from provider.")
                return choices[0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            # If json_object mode was rejected, retry without response_format
            if json_mode and ("response_format" in err_body or "json" in err_body.lower()):
                return self.generate(prompt, system_prompt, json_mode=False)
            raise RuntimeError(f"OpenAI Provider HTTP {e.code}: {err_body}")
        except Exception as e:
            raise RuntimeError(f"Error querying OpenAI-compatible provider: {e}")
