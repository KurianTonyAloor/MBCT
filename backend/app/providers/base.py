"""Abstract base interface for calibration dataset generation providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import re
from app.core.types import ProviderConfig


class BaseCalibrationProvider(ABC):
    """Abstract interface matching the blueprint's CalibrationProvider specification."""

    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config or ProviderConfig()

    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """Validate connection using a minimal request (low token usage)."""
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str, json_mode: bool = True) -> str:
        """Call provider API to generate raw text or JSON response."""
        pass

    def generate_contrastive_pair(self, trait_id: str, scenario: str, system_prompt: str) -> Dict[str, Any]:
        """Standard contrastive pair generation with schema enforcement and robust parsing."""
        user_instruction = f"""Generate 1 contrastive pair for scenario: '{scenario}'.
Return ONLY a valid JSON object matching this schema:
{{
  "scenario": "{scenario}",
  "user_prompt": "realistic user query",
  "target_response": "response exemplifying high {trait_id}",
  "contrast_response": "response exemplifying low {trait_id}",
  "invariant_summary": "explanation of invariant factual claims/answers in both"
}}"""

        raw_output = self.generate(user_instruction, system_prompt, json_mode=True)
        
        # Parse JSON directly or extract using regex
        try:
            return json.loads(raw_output.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Provider failed to produce valid JSON: {raw_output[:250]}")
