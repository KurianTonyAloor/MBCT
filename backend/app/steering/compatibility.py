"""Robust calibration compatibility and fingerprint validation across model revisions and runtimes."""

import hashlib
import torch
from typing import Dict, Any, List, Optional
from app.core.types import (
    ModelCalibrationProfile,
    ModelFingerprint,
    CompatibilityReport
)
from app.steering.hooks import ActivationHookManager


class CompatibilityValidator:
    """Validates whether a Model Calibration Profile (MCP) is compatible with a target runtime model."""

    @classmethod
    def compute_fingerprint(cls, model: Any, tokenizer: Any, model_key: str = "") -> ModelFingerprint:
        """Generate a deterministic fingerprint from a loaded model and tokenizer."""
        config = getattr(model, "config", None)
        arch = model.__class__.__name__
        hidden_size = getattr(
            config, "hidden_size",
            getattr(config, "n_embd", getattr(config, "d_model", 0))
        )
        layer_count = getattr(
            config, "num_hidden_layers",
            getattr(config, "n_layer", getattr(config, "num_layers", 0))
        )
        vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer) if tokenizer else 32000)
        precision = str(getattr(model, "dtype", "unknown"))
        device = str(getattr(model, "device", "cpu"))

        # Compute deterministic hash
        hash_seed = f"{model_key}:{arch}:{hidden_size}:{layer_count}:{vocab_size}:{precision}"
        revision_hash = hashlib.sha256(hash_seed.encode("utf-8")).hexdigest()[:16]

        return ModelFingerprint(
            model_id=model_key or getattr(config, "_name_or_path", "unknown"),
            architecture=arch,
            hidden_size=hidden_size,
            layer_count=layer_count,
            vocab_size=vocab_size,
            precision=precision,
            device=device,
            revision_hash=revision_hash
        )

    @classmethod
    def validate(
        cls,
        mcp: ModelCalibrationProfile,
        model: Any,
        hook_manager: ActivationHookManager,
        model_key: str = ""
    ) -> CompatibilityReport:
        """
        Validate MCP against live model instance.
        Returns CompatibilityReport with is_compatible bool, violations, and warnings.
        """
        violations: List[str] = []
        warnings: List[str] = []

        config = getattr(model, "config", None)
        model_hidden = getattr(
            config, "hidden_size",
            getattr(config, "n_embd", getattr(config, "d_model", 0))
        )
        num_layers = hook_manager.num_layers
        arch = model.__class__.__name__

        # ── 1. Critical Check: Vector Dimension Match ────────────────────────────
        first_vec_len = None
        for trait_id, layer_dict in mcp.vectors.items():
            for l_key, vec in layer_dict.items():
                if vec and len(vec) > 0:
                    first_vec_len = len(vec)
                    break
            if first_vec_len is not None:
                break

        if first_vec_len is not None and model_hidden > 0:
            if first_vec_len != model_hidden:
                violations.append(
                    f"FATAL: Vector hidden dimension mismatch. MCP vector dimension is {first_vec_len}, "
                    f"but model '{arch}' hidden_size is {model_hidden}. "
                    f"Intervention would crash the residual stream."
                )

        # ── 2. Critical Check: Layer Index Bounds ────────────────────────────────
        all_mcp_layers = set()
        for trait_id, layers in mcp.chosen_layers.items():
            for l in layers:
                all_mcp_layers.add(int(l))

        for l in all_mcp_layers:
            if l < 0 or l >= num_layers:
                violations.append(
                    f"FATAL: Layer index {l} in MCP exceeds model layer count {num_layers} (valid: 0 to {num_layers - 1})."
                )

        # ── 3. Architecture Family Check ─────────────────────────────────────────
        mcp_fp = mcp.model_fingerprint.lower()
        active_key = model_key.lower()
        active_arch = arch.lower()

        # Check for cross-family contamination
        families = ["phi", "llama", "qwen", "gpt2", "gemma", "mistral"]
        mcp_family = next((f for f in families if f in mcp_fp or f in mcp.model_id.lower()), None)
        active_family = next((f for f in families if f in active_key or f in active_arch), None)

        if mcp_family and active_family and mcp_family != active_family:
            violations.append(
                f"FATAL: Architecture family mismatch. MCP is calibrated for '{mcp_family}', "
                f"but target model is '{active_family}' ({arch}). Steering vectors are non-transferable across different architectures."
            )
        elif mcp.model_id and model_key and mcp.model_id.lower() != model_key.lower():
            warnings.append(
                f"Model ID difference: MCP was calibrated for '{mcp.model_id}', active model is '{model_key}'. "
                f"Ensure both are equivalent revisions."
            )

        # ── 4. Numerical Vector Sanity Check ─────────────────────────────────────
        has_nans = False
        all_zeros = True
        for trait_id, layer_dict in mcp.vectors.items():
            for l_key, vec in layer_dict.items():
                if vec:
                    for val in vec:
                        if val != 0.0:
                            all_zeros = False
                        if val != val:  # NaN check
                            has_nans = True
                            break

        if has_nans:
            violations.append("FATAL: MCP steering vectors contain NaN values (numerical corruption).")
        if all_zeros and len(mcp.vectors) > 0:
            warnings.append("WARNING: All MCP steering vectors are zero. Steering will have no effect.")

        # Determine overall compatibility
        is_compatible = len(violations) == 0
        if not is_compatible:
            status = "incompatible"
        elif len(warnings) > 0:
            status = "needs_validation"
        else:
            status = "compatible"

        return CompatibilityReport(
            is_compatible=is_compatible,
            status=status,
            target_model=model_key or arch,
            mcp_id=mcp.mcp_id,
            violations=violations,
            warnings=warnings,
            reusable=is_compatible and len(warnings) == 0
        )
