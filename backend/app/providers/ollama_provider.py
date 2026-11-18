"""Calibration provider adapter for local Ollama and OpenAI-compatible endpoints."""

import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from app.core.types import ProviderConfig


class OllamaProvider:
    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config or ProviderConfig()

    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Ollama server with a minimal non-intrusive probe."""
        endpoint = self.config.endpoint.rstrip("/")
        try:
            # First check root/version endpoint
            req = urllib.request.Request(f"{endpoint}/api/tags", headers={"User-Agent": "MBCT/1.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                models = [m.get("name") for m in data.get("models", [])]
                model_exists = any(self.config.model_id in m for m in models)
                return {
                    "success": True,
                    "available_models": models,
                    "target_model_available": model_exists,
                    "message": f"Connected to Ollama. Target model '{self.config.model_id}' available: {model_exists}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Could not reach Ollama at {endpoint}. Ensure 'ollama serve' is running."
            }

    def generate(self, prompt: str, system_prompt: str, json_mode: bool = True) -> str:
        """Call Ollama generation API to produce candidate contrastive data."""
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
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode())
                return result.get("response", "")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            raise RuntimeError(f"Ollama HTTP {e.code}: {err_body}")
        except Exception as e:
            raise RuntimeError(f"Error querying Ollama provider: {e}")

    def generate_contrastive_pair(self, trait_id: str, scenario: str, system_prompt: str) -> Dict[str, Any]:
        """Ask provider to generate one contrastive pair according to the strict blueprint schema."""
        user_instruction = f"""Generate 1 contrastive pair for scenario: '{scenario}'.
Return ONLY a valid JSON object with the following schema:
{{
  "scenario": "{scenario}",
  "user_prompt": "string: realistic user query",
  "target_response": "string: response exemplifying high {trait_id}",
  "contrast_response": "string: response exemplifying low {trait_id}",
  "invariant_summary": "string: explanation of what factual claims/answers are identical in both"
}}"""

        raw_output = self.generate(user_instruction, system_prompt, json_mode=True)
        
        # Robustly parse JSON from output
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            # Fallback regex search for JSON block
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Provider failed to return valid JSON: {raw_output[:200]}")
