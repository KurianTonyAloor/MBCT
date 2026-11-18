"""Activation extraction and contrastive vector derivation engine."""

import torch
from typing import List, Dict, Tuple, Optional
from app.core.types import CalibrationExample, SteeringVectorArtifact
from app.steering.hooks import ActivationHookManager


class VectorDeriver:
    def __init__(self, model, tokenizer, hook_manager: ActivationHookManager):
        self.model = model
        self.tokenizer = tokenizer
        self.hook_manager = hook_manager

    def extract_pair_activations(
        self,
        prompt: str,
        target_resp: str,
        contrast_resp: str,
        target_layers: List[int]
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Teacher-forced forward passes extracting response token activations for a single pair."""
        self.hook_manager.attach_hooks(target_layers)
        
        # 1. Measure prompt token boundary
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_tokens)

        # Helper to run forward pass on prompt + response
        def run_pass(response_text: str) -> Dict[int, torch.Tensor]:
            full_text = prompt + "\n" + response_text
            encoded = self.tokenizer(full_text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.model.device)
            attention_mask = encoded.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)

            total_len = input_ids.shape[1]
            # Response tokens span from prompt_len to total_len
            resp_start = min(prompt_len, total_len - 1)
            resp_end = total_len

            self.hook_manager.set_capture_mode(True, token_range=(resp_start, resp_end))
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            captured = {k: v.clone() for k, v in self.hook_manager.captured_activations.items()}
            self.hook_manager.set_capture_mode(False)
            return captured

        pos_activations = run_pass(target_resp)
        contrast_activations = run_pass(contrast_resp)
        return pos_activations, contrast_activations

    def derive_trait_vectors(
        self,
        trait_id: str,
        examples: List[CalibrationExample],
        target_layers: Optional[List[int]] = None
    ) -> Dict[int, SteeringVectorArtifact]:
        """Derive mean contrastive steering vectors across all examples for a given trait."""
        num_layers = self.hook_manager.num_layers
        if target_layers is None:
            # Default to mid-to-late transformer layers (30% to 80% depth)
            start_l = int(num_layers * 0.3)
            end_l = max(start_l + 1, int(num_layers * 0.8))
            target_layers = list(range(start_l, end_l))

        diff_sums: Dict[int, torch.Tensor] = {}
        sample_count = 0

        for ex in examples:
            if not ex.validation.schema_valid or ex.validation.trait_isolation == "failed":
                continue

            try:
                pos_acts, cont_acts = self.extract_pair_activations(
                    prompt=ex.user_prompt,
                    target_resp=ex.positive_response,
                    contrast_resp=ex.contrast_response,
                    target_layers=target_layers
                )

                for layer_idx in target_layers:
                    if layer_idx in pos_acts and layer_idx in cont_acts:
                        diff = pos_acts[layer_idx] - cont_acts[layer_idx]  # shape: [1, hidden_size]
                        if layer_idx not in diff_sums:
                            diff_sums[layer_idx] = torch.zeros_like(diff)
                        diff_sums[layer_idx] += diff

                sample_count += 1
            except Exception as e:
                print(f"Warning: Extraction failed on example {ex.example_id}: {e}")

        if sample_count == 0:
            raise RuntimeError(f"No valid examples available to extract vectors for trait '{trait_id}'")

        artifacts: Dict[int, SteeringVectorArtifact] = {}
        for layer_idx, diff_sum in diff_sums.items():
            mean_diff = (diff_sum / sample_count).squeeze(0)  # [hidden_size]
            norm = float(torch.norm(mean_diff, p=2).item())
            
            # Unit normalized vector
            unit_vector = mean_diff / max(norm, 1e-8)
            vec_list = unit_vector.tolist()

            artifacts[layer_idx] = SteeringVectorArtifact(
                trait_id=trait_id,
                layer_id=layer_idx,
                norm=round(norm, 4),
                sample_count=sample_count,
                vector=vec_list,
                strength_curve={"min_alpha": -3.0, "default_alpha": 1.5, "max_alpha": 3.5}
            )

        return artifacts
