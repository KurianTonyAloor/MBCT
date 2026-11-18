"""Core domain types and data models matching the engineering blueprint."""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import time


class QualityPreset(str, Enum):
    ECONOMICAL = "economical"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


class JobState(str, Enum):
    QUEUED = "queued"
    GENERATING_DATASET = "generating_dataset"
    VALIDATING_DATASET = "validating_dataset"
    AWAITING_REVIEW = "awaiting_review"
    EXTRACTING_ACTIVATIONS = "extracting_activations"
    DERIVING_VECTORS = "deriving_vectors"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProviderConfig(BaseModel):
    provider_id: str = "ollama-local"
    endpoint: str = "http://127.0.0.1:11434"
    model_id: str = "phi3"
    api_key: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    last_validated_at: Optional[float] = None
    is_valid: bool = False


class ModelDefinition(BaseModel):
    model_id: str
    revision: str = "default"
    architecture: str
    hidden_size: int
    layer_count: int
    target_layers: List[int]
    device: str = "cpu"
    supports_steering: bool = True


class ValidationResult(BaseModel):
    schema_valid: bool = True
    semantic_equivalence: str = "pending"  # "passed", "flagged", "failed", "pending"
    trait_isolation: str = "pending"
    factual_consistency: str = "passed"
    length_confound_ratio: float = 1.0
    notes: List[str] = []


class CalibrationExample(BaseModel):
    example_id: str
    trait_id: str
    intensity: float = 0.8
    domain: str = "general"
    scenario: str
    user_prompt: str
    positive_response: str
    contrast_response: str
    invariant_summary: str
    validation: ValidationResult = Field(default_factory=ValidationResult)


class SteeringVectorArtifact(BaseModel):
    trait_id: str
    layer_id: int
    norm: float
    sample_count: int
    # Vector stored as flat list for serialization or referenced via blob
    vector: Optional[List[float]] = None
    strength_curve: Dict[str, float] = Field(default_factory=dict)


class UniversalBehaviourProfile(BaseModel):
    profile_id: str = "default-assistant"
    schema_version: str = "1.0"
    traits: Dict[str, float] = Field(
        default_factory=lambda: {
            "directness": 0.50,
            "formality": 0.50,
            "verbosity": 0.50,
            "empathy": 0.50,
            "initiative": 0.50,
        }
    )
    preferences: Dict[str, bool] = Field(
        default_factory=lambda: {
            "challenge_assumptions": False,
            "proactive_suggestions": False,
        }
    )
    updated_at: float = Field(default_factory=time.time)


class ModelCalibrationProfile(BaseModel):
    mcp_id: str
    model_id: str
    model_fingerprint: str
    traits: List[str]
    chosen_layers: Dict[str, List[int]]  # trait_id -> list of layer indices
    vectors: Dict[str, Dict[int, List[float]]]  # trait_id -> layer_id -> vector
    strength_curves: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    evaluation_report: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    status: str = "ready"


class CompiledSteeringPlan(BaseModel):
    plan_id: str
    ubp_id: str
    mcp_id: str
    enabled: bool = True
    active_traits: Dict[str, bool] = Field(default_factory=dict)
    resolved_layer_weights: Dict[int, Dict[str, float]] = Field(default_factory=dict)  # layer -> trait -> alpha
    compatibility_warnings: List[str] = Field(default_factory=list)
    compiled_at: float = Field(default_factory=time.time)
