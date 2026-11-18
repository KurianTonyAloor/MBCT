"""PyTorch forward hook manager for residual-stream activation capture and steering."""

import torch
from typing import Dict, List, Optional, Tuple, Callable


class ActivationHookManager:
    """Manages forward hooks for capturing and modifying hidden states in transformer layers."""

    def __init__(self, model: torch.nn.Module, layer_module_path: str = "model.layers"):
        self.model = model
        self.layer_module_path = layer_module_path
        self.layers = self._resolve_layers()
        self.hook_handles = []
        
        # Capture state
        self.captured_activations: Dict[int, torch.Tensor] = {}
        self.capture_mode: bool = False
        self.token_range: Optional[Tuple[int, int]] = None  # (start_idx, end_idx)
        
        # Steering state
        self.steering_enabled: bool = False
        self.steering_vectors: Dict[int, torch.Tensor] = {}  # layer_idx -> vector tensor (hidden_size,)
        self.steering_multipliers: Dict[int, float] = {}    # layer_idx -> alpha

    def _resolve_layers(self) -> List[torch.nn.Module]:
        """Navigate attribute path to find transformer decoder layers."""
        obj = self.model
        for part in self.layer_module_path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                # Fallback heuristics for common architectures
                if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                    return list(self.model.transformer.h)
                elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                    return list(self.model.model.layers)
                raise AttributeError(f"Could not resolve layer path '{self.layer_module_path}' in model")
        return list(obj)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def attach_hooks(self, target_layers: Optional[List[int]] = None):
        """Attach forward hooks to specified layers (or all layers if None)."""
        self.remove_hooks()
        layers_to_hook = target_layers if target_layers is not None else list(range(self.num_layers))

        for layer_idx in layers_to_hook:
            if layer_idx < 0 or layer_idx >= self.num_layers:
                continue
            layer = self.layers[layer_idx]
            handle = layer.register_forward_hook(self._create_hook_fn(layer_idx))
            self.hook_handles.append(handle)

    def remove_hooks(self):
        """Detach all hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def _create_hook_fn(self, layer_idx: int) -> Callable:
        def hook_fn(module, input, output):
            # Transformer layer output is typically (hidden_states, present_key_value, ...)
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output

            # 1. Capture mode: record hidden states at selected token positions
            if self.capture_mode:
                with torch.no_grad():
                    # hidden_states shape: [batch_size, seq_len, hidden_size]
                    if self.token_range is not None:
                        start_pos, end_pos = self.token_range
                        # Mean pool across response token positions
                        selected = hidden_states[:, start_pos:end_pos, :].detach().cpu()
                        pooled = selected.mean(dim=1)  # [batch_size, hidden_size]
                    else:
                        pooled = hidden_states.detach().cpu().mean(dim=1)
                    self.captured_activations[layer_idx] = pooled

            # 2. Steering mode: inject steering vector into residual stream
            if self.steering_enabled and layer_idx in self.steering_vectors:
                vec = self.steering_vectors[layer_idx].to(hidden_states.device, dtype=hidden_states.dtype)
                alpha = self.steering_multipliers.get(layer_idx, 1.0)
                if alpha != 0.0:
                    # Inject vector into the sequence positions
                    # Broadcast vector [hidden_size] to [batch, seq_len, hidden_size]
                    delta = (alpha * vec).unsqueeze(0).unsqueeze(0)
                    hidden_states = hidden_states + delta

            if is_tuple:
                return (hidden_states,) + output[1:]
            return hidden_states

        return hook_fn

    def set_capture_mode(self, enabled: bool, token_range: Optional[Tuple[int, int]] = None):
        """Enable or disable activation capture."""
        self.capture_mode = enabled
        self.token_range = token_range
        if not enabled:
            self.captured_activations.clear()

    def configure_steering(
        self,
        steering_vectors: Dict[int, torch.Tensor],
        multipliers: Dict[int, float],
        enabled: bool = True
    ):
        """Update active steering vectors, layer multipliers, and master toggle."""
        self.steering_vectors = steering_vectors
        self.steering_multipliers = multipliers
        self.steering_enabled = enabled

    def set_steering_enabled(self, enabled: bool):
        """Master kill switch: instantly enable or disable all steering interventions."""
        self.steering_enabled = enabled
