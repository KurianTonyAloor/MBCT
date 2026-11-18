"""Personality Compiler: maps Universal Behaviour Profile (UBP) to MCP layer injection weights.
Phase 4 addition: pairwise cosine-similarity check + optional vector orthogonalization.
"""

import uuid
import math
from typing import Dict, List, Optional
from app.core.types import (
    UniversalBehaviourProfile,
    ModelCalibrationProfile,
    CompiledSteeringPlan
)


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(_dot(v, v))


def _cosine(a: List[float], b: List[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return _dot(a, b) / (na * nb)


def _gram_schmidt_project_out(vec: List[float], basis: List[float]) -> List[float]:
    """Remove the component of `vec` along `basis` (Gram-Schmidt step)."""
    nb = _norm(basis)
    if nb < 1e-9:
        return vec
    proj_coeff = _dot(vec, basis) / (nb * nb)
    return [v - proj_coeff * b for v, b in zip(vec, basis)]


class PersonalityCompiler:
    # Cosine-similarity threshold above which a warning (or orthogonalization) triggers
    SIMILARITY_WARN_THRESHOLD = 0.60
    SIMILARITY_ORTHO_THRESHOLD = 0.75  # Above this, automatically orthogonalize

    @staticmethod
    def compile(
        ubp: UniversalBehaviourProfile,
        mcp: ModelCalibrationProfile,
        active_trait_toggles: Optional[Dict[str, bool]] = None,
        orthogonalize: bool = True
    ) -> CompiledSteeringPlan:
        """Compile model-independent slider values into model-specific layer injection weights.

        Steps:
          1. Map each UBP slider → alpha per layer (centered at 0.5 neutral).
          2. Check pairwise cosine similarity of steering vectors at each layer.
          3. If similarity > ORTHO threshold: project out the shared direction.
          4. Store warnings for all pairs above WARN threshold.
        """
        active_toggles = active_trait_toggles or {}
        resolved_weights: Dict[int, Dict[str, float]] = {}
        warnings: List[str] = []

        # Default scale for unit steering vectors
        DEFAULT_BASE_SCALE = 2.5

        # ── Step 1: Compute alphas ───────────────────────────────────────────
        trait_alphas: Dict[str, float] = {}
        for trait_id, slider_val in ubp.traits.items():
            if active_toggles.get(trait_id, True) is False:
                continue
            if trait_id not in mcp.vectors:
                warnings.append(f"Trait '{trait_id}' has no calibrated vectors in MCP '{mcp.mcp_id}'.")
                continue

            centered = (slider_val - 0.5) * 2.0  # −1 to +1
            trait_curve = mcp.strength_curves.get(trait_id, {})
            max_alpha = trait_curve.get("max_alpha", DEFAULT_BASE_SCALE)
            alpha = centered * max_alpha

            if abs(alpha) < 1e-4:
                continue  # Neutral slider → no injection
            trait_alphas[trait_id] = alpha

        active_traits = list(trait_alphas.keys())

        # ── Step 2: Per-layer cosine similarity check & optional ortho ───────
        # Collect all layers across active traits
        all_layers: set = set()
        for trait_id in active_traits:
            layers = mcp.chosen_layers.get(trait_id, list(mcp.vectors[trait_id].keys()))
            all_layers.update(int(l) for l in layers)

        # For each layer, check pairwise similarities and optionally orthogonalize
        orthogonalized_vecs: Dict[str, Dict[int, List[float]]] = {
            t: {} for t in active_traits
        }
        # Initialize with stored vectors
        for trait_id in active_traits:
            for layer_raw, vec in mcp.vectors[trait_id].items():
                layer_idx = int(layer_raw)
                orthogonalized_vecs[trait_id][layer_idx] = list(vec)

        # Pairwise check at each layer
        for layer_idx in sorted(all_layers):
            for i, t1 in enumerate(active_traits):
                for t2 in active_traits[i + 1:]:
                    v1 = orthogonalized_vecs[t1].get(layer_idx)
                    v2 = orthogonalized_vecs[t2].get(layer_idx)
                    if v1 is None or v2 is None:
                        continue
                    sim = _cosine(v1, v2)
                    if abs(sim) >= PersonalityCompiler.SIMILARITY_WARN_THRESHOLD:
                        msg = (
                            f"Layer {layer_idx}: traits '{t1}' and '{t2}' have "
                            f"cosine similarity {sim:.3f}"
                        )
                        if abs(sim) >= PersonalityCompiler.SIMILARITY_ORTHO_THRESHOLD and orthogonalize:
                            # Project t2's vector out of t1's direction
                            new_v2 = _gram_schmidt_project_out(v2, v1)
                            new_norm = _norm(new_v2)
                            if new_norm > 1e-6:
                                # Re-normalize to preserve original magnitude
                                orig_norm = _norm(v2)
                                scale = orig_norm / new_norm
                                new_v2 = [x * scale for x in new_v2]
                                orthogonalized_vecs[t2][layer_idx] = new_v2
                                warnings.append(msg + f" → orthogonalized '{t2}' at layer {layer_idx}.")
                            else:
                                warnings.append(msg + f" (WARNING: '{t2}' nearly parallel to '{t1}', orthogonalization removed it).")
                        else:
                            warnings.append(msg + " (above similarity threshold — consider orthogonalization).")

        # ── Step 3: Build resolved_layer_weights using (possibly ortho'd) vecs ─
        # Note: the layer weights dict stores ALPHAS not vectors; actual vector
        # injection in controller.py uses mcp.vectors. We need to store ortho'd
        # vectors separately — but for now we store them in the plan as metadata
        # and pass them to controller via apply_plan.
        for trait_id in active_traits:
            alpha = trait_alphas[trait_id]
            layers = mcp.chosen_layers.get(trait_id, list(mcp.vectors[trait_id].keys()))
            for layer_raw in layers:
                layer_idx = int(layer_raw)
                if layer_idx not in resolved_weights:
                    resolved_weights[layer_idx] = {}
                resolved_weights[layer_idx][trait_id] = round(alpha, 4)

        plan = CompiledSteeringPlan(
            plan_id=f"plan-{uuid.uuid4().hex[:8]}",
            ubp_id=ubp.profile_id,
            mcp_id=mcp.mcp_id,
            enabled=True,
            active_traits={t: True for t in active_traits},
            resolved_layer_weights=resolved_weights,
            compatibility_warnings=warnings
        )
        # Attach orthogonalized vectors as runtime metadata (used by controller)
        plan.__dict__["_orthogonalized_vecs"] = orthogonalized_vecs
        return plan
