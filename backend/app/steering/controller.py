"""Runtime steering controller that executes generation with compiled activation interventions."""

import torch
from typing import Dict, List, Optional, Iterator
from transformers import TextIteratorStreamer
from threading import Thread
from app.core.types import CompiledSteeringPlan, ModelCalibrationProfile, CompatibilityReport
from app.steering.hooks import ActivationHookManager
from app.steering.compatibility import CompatibilityValidator
from app.steering.layer_budget import select_injection_layers


class SteeringController:
    def __init__(self, model, tokenizer, hook_manager: ActivationHookManager):
        self.model = model
        self.tokenizer = tokenizer
        self.hook_manager = hook_manager
        self.current_plan: Optional[CompiledSteeringPlan] = None
        self.current_mcp: Optional[ModelCalibrationProfile] = None
        self.master_enabled: bool = True

    def set_master_enabled(self, enabled: bool):
        """Global kill-switch controlling steering intervention."""
        self.master_enabled = enabled
        self.hook_manager.set_steering_enabled(enabled)

    def apply_plan(self, plan: CompiledSteeringPlan, mcp: ModelCalibrationProfile):
        """Prepare hook manager with compiled layer weights and steering vectors.
        Validates model runtime and revision compatibility before applying interventions.
        Uses orthogonalized vectors from plan._orthogonalized_vecs if available (Phase 4).
        """
        self.current_plan = plan
        self.current_mcp = mcp

        if not plan.enabled or not self.master_enabled:
            self.hook_manager.set_steering_enabled(False)
            return

        # ── Robust Calibration Compatibility Validation (Blueprint Section 13) ──
        compat_report = CompatibilityValidator.validate(mcp, self.model, self.hook_manager)
        if not compat_report.is_compatible:
            plan.compatibility_warnings.extend(compat_report.violations)
            self.hook_manager.set_steering_enabled(False)
            print(f"[SteeringController] ERROR: REJECTED MCP '{mcp.mcp_id}' due to compatibility violations: {compat_report.violations}", flush=True)
            return

        if compat_report.warnings:
            plan.compatibility_warnings.extend(compat_report.warnings)

        # Use orthogonalized vectors if compiler produced them
        ortho_vecs = getattr(plan, "_orthogonalized_vecs", None) or \
                     plan.__dict__.get("_orthogonalized_vecs", None)

        combined_vectors: Dict[int, torch.Tensor] = {}
        multipliers: Dict[int, float] = {}

        for layer_idx_raw, trait_alphas in plan.resolved_layer_weights.items():
            layer_idx = int(layer_idx_raw)
            composite_vec = None
            total_weight = 0.0

            for trait_id, alpha in trait_alphas.items():
                # Prefer orthogonalized vector; fall back to raw MCP vector
                if ortho_vecs and trait_id in ortho_vecs:
                    raw_vec = ortho_vecs[trait_id].get(layer_idx)
                else:
                    raw_vec = None

                if raw_vec is None and trait_id in mcp.vectors:
                    trait_dict = mcp.vectors[trait_id]
                    raw_vec = trait_dict.get(layer_idx) or trait_dict.get(str(layer_idx))

                if raw_vec is not None:
                    vec_tensor = torch.tensor(raw_vec, dtype=self.model.dtype, device=self.model.device)
                    if composite_vec is None:
                        composite_vec = alpha * vec_tensor
                    else:
                        composite_vec = composite_vec + (alpha * vec_tensor)
                    total_weight += abs(alpha)

            if composite_vec is not None and total_weight > 0:
                # ── Magnitude normalisation ──────────────────────────────────
                # Unit trait vectors summed across traits grow as sqrt(sum(alpha^2)),
                # so a 3-trait preset injects far harder than a 1-trait preset and
                # over-steers the model (it starts reciting context instead of
                # answering). Keep the composite direction but clamp its length to
                # the strongest single trait's alpha.
                composite_norm = float(composite_vec.norm().item())
                dominant_alpha = max(abs(a) for a in trait_alphas.values())
                if composite_norm > 1e-8 and dominant_alpha > 0.0 and composite_norm > dominant_alpha:
                    composite_vec = composite_vec * (dominant_alpha / composite_norm)
                combined_vectors[layer_idx] = composite_vec
                multipliers[layer_idx] = 1.0

        # ── Model-size guard (defence in depth) ──────────────────────────────
        # A plan compiled without a layer count still cannot exceed what the
        # loaded model can take: every extra hooked layer costs a Python
        # callback on every forward pass and compounds the injected direction.
        if combined_vectors:
            kept, budget_notes = select_injection_layers(
                sorted(combined_vectors), self.hook_manager.num_layers
            )
            if budget_notes:
                plan.compatibility_warnings.extend(budget_notes)
            combined_vectors = {l: combined_vectors[l] for l in kept}
            multipliers = {l: v for l, v in multipliers.items() if l in combined_vectors}

        target_layers = list(combined_vectors.keys())
        if target_layers:
            self.hook_manager.attach_hooks(target_layers)
            self.hook_manager.configure_steering(
                steering_vectors=combined_vectors,
                multipliers=multipliers,
                enabled=True
            )
            warnings = getattr(plan, "compatibility_warnings", [])
            warn_str = f" [WARNING: {len(warnings)} ortho warnings]" if warnings else ""
            print(f"Steering configured on layers {target_layers} with combined vectors.{warn_str}", flush=True)
        else:
            self.hook_manager.set_steering_enabled(False)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        temperature: float = 0.0,
        top_p: float = 0.9,
        steered: bool = True
    ) -> str:
        """Generate response tokens with optional activation steering."""
        should_steer = (
            steered
            and self.master_enabled
            and (self.current_plan is not None and self.current_plan.enabled)
        )
        self.hook_manager.set_steering_enabled(should_steer)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "use_cache": True,  # KV cache for 3x-5x faster autoregressive decoding on CPU
                    "pad_token_id": self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
                }
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = top_p
                    gen_kwargs["do_sample"] = True
                else:
                    gen_kwargs["do_sample"] = False

                outputs = self.model.generate(**inputs, **gen_kwargs)

            new_tokens = outputs[0][prompt_len:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return response
        finally:
            self.hook_manager.set_steering_enabled(self.master_enabled)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.0,
        top_p: float = 0.9,
        steered: bool = True
    ) -> Iterator[str]:
        """Generate tokens one at a time using TextIteratorStreamer (runs in background thread)."""
        should_steer = (
            steered
            and self.master_enabled
            and (self.current_plan is not None and self.current_plan.enabled)
        )
        self.hook_manager.set_steering_enabled(should_steer)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        def _run():
            try:
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)
            finally:
                self.hook_manager.set_steering_enabled(self.master_enabled)

        thread = Thread(target=_run, daemon=True)
        thread.start()

        # Yield tokens as they arrive from the streamer
        for token_text in streamer:
            yield token_text

        thread.join(timeout=120)
