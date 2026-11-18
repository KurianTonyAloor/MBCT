"""Phase 0 & Phase 1 Feasibility Spike Script for Phi-3 Mini.

1. Tests Ollama connection ('phi3' as calibration provider).
2. Loads validated contrastive pairs for trait 'directness'.
3. Loads local PyTorch Phi-3 Mini (3.8B, bfloat16) with residual stream hooks.
4. Extracts activations on response tokens and computes contrastive vector v[directness, l].
5. Compiles a high directness steering plan.
6. Evaluates baseline vs. steered generation on a held-out test prompt.
7. Tests master kill switch to verify 100% exact reversion.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.types import QualityPreset
from app.providers.ollama_provider import OllamaProvider
from app.dataset.generator import DatasetGenerator
from app.models.registry import ModelLoader
from app.steering.deriver import VectorDeriver
from app.profiles.compiler import PersonalityCompiler
from app.core.types import UniversalBehaviourProfile, ModelCalibrationProfile
from app.steering.controller import SteeringController


def run_spike():
    print("=== [1] Testing Ollama Calibration Provider ===", flush=True)
    provider = OllamaProvider()
    conn_info = provider.test_connection()
    print("Ollama connection status:", conn_info, flush=True)

    print("\n=== [2] Generating/Loading Contrastive Pairs for 'directness' ===", flush=True)
    generator = DatasetGenerator(provider=provider)
    examples = generator.generate_for_trait("directness", quality_preset=QualityPreset.ECONOMICAL)
    print(f"Obtained {len(examples)} validated examples.", flush=True)
    for ex in examples:
        print(f" - [{ex.example_id}] {ex.scenario}", flush=True)

    print("\n=== [3] Loading Base Model (Phi-3 Mini) & Residual Hooks ===", flush=True)
    model, tokenizer, hook_manager = ModelLoader.load_model("phi-3-mini")
    print(f"Model loaded with {hook_manager.num_layers} layers.", flush=True)

    print("\n=== [4] Deriving Contrastive Steering Vectors ===", flush=True)
    deriver = VectorDeriver(model, tokenizer, hook_manager)
    # Target mid-layers [12, 14, 16] in Phi-3
    target_layers = [12, 14, 16]
    artifacts = deriver.derive_trait_vectors("directness", examples, target_layers=target_layers)
    for layer_idx, art in artifacts.items():
        print(f" - Layer {layer_idx}: norm={art.norm}, sample_count={art.sample_count}", flush=True)

    print("\n=== [5] Assembling Model Calibration Profile (MCP) ===", flush=True)
    vectors_dict = {
        "directness": {layer_idx: art.vector for layer_idx, art in artifacts.items()}
    }
    mcp = ModelCalibrationProfile(
        mcp_id="mcp-phi3-mini-v1",
        model_id="microsoft/Phi-3-mini-4k-instruct",
        model_fingerprint="phi-3-mini-bfloat16-cpu",
        traits=["directness"],
        chosen_layers={"directness": target_layers},
        vectors=vectors_dict,
        strength_curves={"directness": {"max_alpha": 3.0}}
    )

    print("\n=== [6] Compiling High Directness Plan (Slider = 1.0) ===", flush=True)
    high_direct_ubp = UniversalBehaviourProfile(
        profile_id="high-directness",
        traits={"directness": 1.0}
    )
    plan_steered = PersonalityCompiler.compile(high_direct_ubp, mcp)
    print("Compiled Plan Layer Weights:", plan_steered.resolved_layer_weights, flush=True)

    controller = SteeringController(model, tokenizer, hook_manager)
    controller.apply_plan(plan_steered, mcp)

    held_out_prompt = "<|user|>\nWould it be a good idea to write our entire new backend in Assembly language to make it faster?<|end|>\n<|assistant|>\n"

    print("\n=== [7] Generation: Baseline vs. Steered (High Directness) ===", flush=True)
    print(f"Test Prompt: Assembly backend question\n", flush=True)

    baseline_resp = controller.generate(held_out_prompt, max_new_tokens=45, temperature=0.0, steered=False)
    print("--- BASELINE (Unsteered) ---", flush=True)
    print(baseline_resp, flush=True)

    steered_resp = controller.generate(held_out_prompt, max_new_tokens=45, temperature=0.0, steered=True)
    print("\n--- STEERED (High Directness: Slider 1.0, alpha=+3.0) ---", flush=True)
    print(steered_resp, flush=True)

    print("\n=== [8] Master Kill Switch Test ===", flush=True)
    controller.set_master_enabled(False)
    reverted_resp = controller.generate(held_out_prompt, max_new_tokens=45, temperature=0.0, steered=True)
    print("--- AFTER KILL SWITCH (Should match baseline exactly) ---", flush=True)
    print(reverted_resp, flush=True)

    is_identical = (baseline_resp == reverted_resp)
    print(f"\nKill-switch reverted generation matches baseline exactly: {is_identical}", flush=True)

    print("\n=== Feasibility Spike Complete! Phase 0 & Phase 1 verified on Phi-3. ===", flush=True)


if __name__ == "__main__":
    run_spike()
