"""PyTorch forward hook manager for residual-stream activation capture and steering."""

import torch
from typing import Dict, List, Optional, Tuple, Callable


class ActivationHookManager:
    """Manages forward hooks for capturing and modifying hidden states in transformer layers.
    Supports dynamic introspection of any HuggingFace or PyTorch transformer architecture.
    """

    def __init__(self, model: torch.nn.Module, layer_module_path: Optional[str] = None):
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
        # layer_idx -> pre-scaled, device/dtype-ready delta of shape [1, 1, hidden_size]
        self.steering_deltas: Dict[int, torch.Tensor] = {}

    def _resolve_layers(self) -> List[torch.nn.Module]:
        """Dynamically resolve transformer decoder layers across any architecture."""
        # 1. Try explicit path if provided
        if self.layer_module_path:
            try:
                obj = self.model
                for part in self.layer_module_path.split("."):
                    obj = getattr(obj, part)
                layers = list(obj)
                if len(layers) > 0 and isinstance(layers[0], torch.nn.Module):
                    return layers
            except Exception:
                pass

        # 2. Common architectural paths across open-source models:
        # - model.layers: LLaMA, Mistral, Qwen, Gemma, Phi-3, Starcoder2, InternLM, Baichuan, DeepSeek
        # - transformer.h: GPT-2, Falcon, Bloom, StarCoder
        # - model.decoder.layers: OPT, BART, Marian
        # - gpt_neox.layers: Pythia, GPT-NeoX, StableLM
        # - transformer.layers: MPT
        candidate_paths = [
            "model.layers",
            "transformer.h",
            "model.decoder.layers",
            "gpt_neox.layers",
            "transformer.layers",
            "encoder.layer",
            "decoder.layers"
        ]
        for path in candidate_paths:
            try:
                obj = self.model
                for part in path.split("."):
                    obj = getattr(obj, part)
                layers = list(obj)
                if len(layers) > 0 and isinstance(layers[0], torch.nn.Module):
                    self.layer_module_path = path
                    return layers
            except Exception:
                continue

        # 3. Heuristic search: find any torch.nn.ModuleList whose children look like transformer layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) >= 4:
                # Check if first element has self_attn or attention or mlp or feed_forward
                first = module[0]
                child_names = [n.lower() for n, _ in first.named_children()]
                if any("attn" in n or "attention" in n for n in child_names):
                    self.layer_module_path = name
                    return list(module)

        raise AttributeError(
            f"Could not automatically resolve transformer layer blocks in model of type {type(self.model).__name__}. "
            f"Please specify layer_module_path explicitly."
        )

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
                # Deltas are pre-scaled and pre-cast in configure_steering, so a
                # decode step costs one broadcast add per hooked layer instead of
                # a multiply, an unsqueeze and two device casts.
                delta = self.steering_deltas.get(layer_idx)
                if delta is None or delta.shape[-1] != hidden_states.shape[-1]:
                    vec = self.steering_vectors[layer_idx].to(hidden_states.device, dtype=hidden_states.dtype)
                    alpha = self.steering_multipliers.get(layer_idx, 1.0)
                    if alpha == 0.0:
                        delta = None
                    else:
                        delta = (alpha * vec).unsqueeze(0).unsqueeze(0)
                if delta is not None:
                    if hidden_states.shape[1] > 1:
                        # Prefill pass: the prompt tokens encode the instruction.
                        # Perturbing them is what makes the model ignore the question,
                        # so steer only the last position (which produces the first
                        # generated token). Decode steps (seq_len == 1) are steered fully.
                        patched = hidden_states.clone()
                        patched[:, -1:, :] = patched[:, -1:, :] + delta
                        hidden_states = patched
                    else:
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
        self._refresh_deltas()

    def _refresh_deltas(self):
        """Cache alpha * vector in the model's device/dtype, once per plan."""
        self.steering_deltas = {}
        try:
            param = next(self.model.parameters())
        except (StopIteration, AttributeError):
            return  # no parameters (stub model): the hook falls back to lazy casting
        for layer_idx, vec in self.steering_vectors.items():
            alpha = self.steering_multipliers.get(layer_idx, 1.0)
            if alpha != 0.0:
                self.steering_deltas[layer_idx] = (alpha * vec).to(
                    param.device, dtype=param.dtype
                ).unsqueeze(0).unsqueeze(0)

    def set_steering_enabled(self, enabled: bool):
        """Master kill switch: instantly enable or disable all steering interventions."""
        self.steering_enabled = enabled
