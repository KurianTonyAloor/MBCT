"""Model-size-aware budget for residual-stream layer injection.

Every steered layer adds its vector into the residual stream of that layer, so
the interventions compound with depth. Five layers on phi-3-mini (32 layers,
~3.8B) is a mild nudge at ~16% of the network; the same five layers on GPT-2
(12 layers) would rewrite ~42% of the depth with the same direction, which
degrades generation. Each hooked layer also costs one Python callback per
forward pass, so wide injection shows up as extra per-token latency on CPU.

The budget therefore follows the model's depth (DEPTH_FRACTION with an
absolute cap) and stays inside the middle band of the network, where
behavioural features are most separable from surface syntax.
"""

from typing import Iterable, List, Tuple

# Share of a model's depth that may carry steering vectors.
DEPTH_FRACTION = 0.15
# Hard ceiling: above ~5 layers the injected directions start to compound into
# degenerate output regardless of how deep the model is.
MAX_LAYERS = 5
MIN_LAYERS = 1
# Fractional depth band (0.0 = first layer, 1.0 = last layer) that is safe to
# perturb: the earliest layers encode tokens, the last ones fix the next token.
BAND_START = 0.28
BAND_END = 0.72


def layer_budget(num_layers: int) -> int:
    """How many layers a model of this depth may be steered on."""
    if num_layers is None or num_layers <= 0:
        return MAX_LAYERS
    return max(MIN_LAYERS, min(MAX_LAYERS, int(round(num_layers * DEPTH_FRACTION))))


def _band(num_layers: int) -> Tuple[int, int]:
    last = max(num_layers - 1, 0)
    lo = int(round(BAND_START * last))
    hi = int(round(BAND_END * last))
    return lo, max(lo, hi)


def _spread(items: List[int], k: int) -> List[int]:
    """Pick k layers spaced evenly across the sorted candidate list."""
    if k >= len(items):
        return list(items)
    if k <= 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (k - 1)
    return [items[int(round(i * step))] for i in range(k)]


def select_injection_layers(
    candidates: Iterable[int],
    num_layers: int,
    budget: int = None,
) -> Tuple[List[int], List[str]]:
    """Trim candidate layer indices to what this model's size can take.

    Returns the selected layers and human-readable notes explaining any trim,
    so callers can surface them as compatibility warnings.
    """
    notes: List[str] = []
    cands = sorted({int(c) for c in candidates})

    if num_layers is not None and num_layers > 0:
        out_of_range = [c for c in cands if c < 0 or c >= num_layers]
        if out_of_range:
            notes.append(f"Layer {out_of_range} outside model depth 0-{num_layers - 1}; skipped.")
            cands = [c for c in cands if 0 <= c < num_layers]

        lo, hi = _band(num_layers)
        outside = [c for c in cands if c < lo or c > hi]
        if outside and len(outside) < len(cands):
            notes.append(
                f"Model size guard: layers {outside} lie outside the steerable band "
                f"{lo}-{hi} of a {num_layers}-layer model; skipped."
            )
            cands = [c for c in cands if lo <= c <= hi]

    limit = budget if budget is not None else layer_budget(num_layers)
    if len(cands) > limit:
        kept = _spread(cands, limit)
        dropped = [c for c in cands if c not in kept]
        notes.append(
            f"Model size guard: {num_layers or 'unknown'} layers allow {limit} injection "
            f"layers ({int(DEPTH_FRACTION * 100)}% of depth, max {MAX_LAYERS}); "
            f"kept {kept}, dropped {dropped}."
        )
        cands = kept

    return cands, notes
