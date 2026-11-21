"""Held-out 300-pair evaluation of every behaviour profile on the active MCP.

Protocol (per pair, per profile):
  * Steered likelihood: teacher-force the trait-positive response under the compiled plan.
  * Baseline likelihood: identical forward pass with the intervention switched off.
  * Delta = mean per-token log-probability difference (nats/token), so it is length-normalised.

Intervention placement matches runtime generation: the last prompt position and every
response position receive h <- h + alpha*v (in generation the prompt is steered once, then
every decode step is steered, and each steered token feeds the next step).

Usage:
  python evaluate_profiles.py --limit 8      # smoke test
  python evaluate_profiles.py                # full 300 pairs x 4 profiles
  python evaluate_profiles.py --analyze      # print the summary table for a finished run
"""

import os
import sys
import json
import time
import math
import argparse
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from app.models.registry import ModelLoader
from app.core.types import ModelCalibrationProfile, UniversalBehaviourProfile
from app.profiles.compiler import PersonalityCompiler
from app.steering.controller import SteeringController
from app.steering.hooks import ActivationHookManager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rooted(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)

PROFILES: Dict[str, Dict[str, float]] = {
    "Direct Architect": {"directness": 0.95, "formality": 0.80, "verbosity": 0.30},
    "Friendly Mentor": {"directness": 0.30, "formality": 0.20, "verbosity": 0.75, "empathy": 0.90},
    "Brutal Reviewer": {"directness": 1.00, "formality": 0.50, "verbosity": 0.25},
    "Neutral Baseline": {"directness": 0.50, "formality": 0.50, "verbosity": 0.50},
}

RESPONSE_TOKENS = 8  # scored prefix length per response

CONTROL_TEXT = (
    "The team met on Tuesday to review the quarterly numbers and agreed to revisit the "
    "hiring plan in the spring."
)

STATE: Dict[str, Any] = {"delta_by_layer": None, "plen": 0}
HOOKS: List[Any] = []


def make_hook(layer_id: int):
    def hook_fn(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        table = STATE["delta_by_layer"]
        if table:
            delta = table.get(layer_id)
            if delta is not None:
                mask = torch.zeros_like(h)
                start = max(STATE["plen"] - 1, 0)
                mask[:, start:, :] = delta
                h = h + mask
        if is_tuple:
            return (h,) + out[1:]
        return h

    return hook_fn


def attach_eval_hooks(hm: ActivationHookManager, layer_ids: List[int]):
    for handle in HOOKS:
        handle.remove()
    HOOKS.clear()
    for lid in layer_ids:
        HOOKS.append(hm.layers[lid].register_forward_hook(make_hook(lid)))


def build_prompt(tok, user_prompt: str) -> List[int]:
    text = ModelLoader.format_prompt([{"role": "user", "content": user_prompt}], tok, "phi-3-mini")
    return tok(text, add_special_tokens=False)["input_ids"]


def score_sequence(model, tok, prompt_ids: List[int], response_text: str) -> Optional[Dict[str, Any]]:
    resp_ids = tok(response_text, add_special_tokens=False)["input_ids"][:RESPONSE_TOKENS]
    if len(resp_ids) < 2:
        return None
    ids = torch.tensor([prompt_ids + resp_ids])
    plen = len(prompt_ids)
    STATE["plen"] = plen
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
    lp = F.log_softmax(logits[plen - 1: len(prompt_ids) + len(resp_ids) - 1].float(), dim=-1)
    tgt = torch.tensor(resp_ids)
    tok_lp = lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    return {
        "mean_logp": float(tok_lp.mean()),
        "n_tokens": int(len(resp_ids)),
        "min_logp": float(tok_lp.min()),
    }


def alpha_for(profile: Dict[str, float], trait_id: str, max_alpha: float) -> float:
    slider = profile.get(trait_id, 0.5)
    return round((slider - 0.5) * 2.0 * max_alpha, 4)


def sign_test_p(wins: int, n: int) -> float:
    """Two-sided exact binomial p-value against p=0.5."""
    if n == 0:
        return 1.0
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def run(args):
    # 8 threads beats the 12 logical CPUs on this box: the pass is memory-bandwidth
    # bound, so the extra hyperthreads only add contention (0.17 vs 0.22 s/token).
    torch.set_num_threads(8)

    with open(rooted(args.pairs), "r", encoding="utf-8") as f:
        pairs: List[Dict[str, Any]] = json.load(f)
    if args.limit:
        by_trait: Dict[str, List[Dict[str, Any]]] = {}
        for p in pairs:
            by_trait.setdefault(p["trait_id"], []).append(p)
        mixed = []
        for i in range(max(len(v) for v in by_trait.values())):
            for t in by_trait:
                if i < len(by_trait[t]):
                    mixed.append(by_trait[t][i])
        pairs = mixed[: args.limit]

    with open(rooted("artifacts/profiles/mcp_phi-3-mini.json"), "r", encoding="utf-8") as f:
        mcp = ModelCalibrationProfile(**json.load(f))

    model, tok, hm = ModelLoader.load_model("phi-3-mini")
    controller = SteeringController(model, tok, hm)
    max_alpha = mcp.strength_curves.get("directness", {}).get("max_alpha", 3.0)

    # ── Compile every profile once, keep its per-layer deltas ────────────────
    plan_deltas: Dict[str, Dict[int, torch.Tensor]] = {}
    plan_layers: Dict[str, List[int]] = {}
    plan_alphas: Dict[str, Dict[str, float]] = {}
    for name, sliders in PROFILES.items():
        ubp = UniversalBehaviourProfile(profile_id=name.lower().replace(" ", "-"), traits=sliders)
        plan = PersonalityCompiler.compile(ubp, mcp, num_layers=hm.num_layers)
        hm.steering_deltas = {}  # apply_plan leaves stale deltas when a plan injects nothing
        controller.apply_plan(plan, mcp)
        deltas = {int(k): v.clone() for k, v in hm.steering_deltas.items()}
        layers = sorted(deltas)
        hm.remove_hooks()
        hm.set_steering_enabled(False)
        plan_deltas[name] = deltas
        plan_layers[name] = layers
        plan_alphas[name] = {t: alpha_for(sliders, t, max_alpha) for t in ("directness", "formality", "verbosity")}
        print(f"[plan] {name}: layers={layers} alphas={plan_alphas[name]}", flush=True)
        if not deltas and name != "Neutral Baseline":
            print(f"[plan] WARNING: {name} produced no steering deltas", flush=True)

    union_layers = sorted({l for ls in plan_layers.values() for l in ls})
    attach_eval_hooks(hm, union_layers)
    print(f"[plan] eval hooks on layers {union_layers}", flush=True)

    control_prompt = build_prompt(tok, "Summarise the following in one sentence.")
    # Neutral Baseline injects nothing, so it is bit-identical to the baseline arm
    # (verified exactly 0.0000 on the smoke set); scoring it as well would only
    # re-measure the control, so it is carried as the identity reference instead.
    evaluated = [p for p in PROFILES if p != "Neutral Baseline"]
    conditions = ["baseline"] + evaluated

    control: Dict[str, float] = {}
    for cond in conditions:
        STATE["delta_by_layer"] = None if cond == "baseline" else plan_deltas[cond]
        sc = score_sequence(model, tok, control_prompt, CONTROL_TEXT)
        control[cond] = sc["mean_logp"] if sc else float("nan")
    print(f"[control] {json.dumps({k: round(v, 4) for k, v in control.items()})}", flush=True)

    results: List[Dict[str, Any]] = []
    t_start = time.time()

    for idx, pair in enumerate(pairs, 1):
        prompt_ids = build_prompt(tok, pair["user_prompt"])
        row: Dict[str, Any] = {
            "example_id": pair["example_id"],
            "trait_id": pair["trait_id"],
            "n_prompt_tokens": len(prompt_ids),
            "scores": {},
        }
        for cond in conditions:
            STATE["delta_by_layer"] = None if cond == "baseline" else plan_deltas[cond]
            sc = score_sequence(model, tok, prompt_ids, pair["positive_response"])
            if sc is not None:
                row["scores"][cond] = sc["mean_logp"]
                row.setdefault("n_tokens", sc["n_tokens"])
        results.append(row)

        if idx % 5 == 0 or idx == len(pairs):
            elapsed = time.time() - t_start
            eta = elapsed / idx * (len(pairs) - idx)
            print(f"[eval] {idx}/{len(pairs)} pairs  elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m", flush=True)
            with open(rooted(args.out), "w", encoding="utf-8") as f:
                json.dump({
                    "pairs": results,
                    "control": control,
                    "config": {
                        "response_tokens": RESPONSE_TOKENS,
                        "alphas": plan_alphas,
                        "layers": plan_layers,
                        "mcp_id": mcp.mcp_id,
                        "profiles": PROFILES,
                        "evaluated": evaluated,
                        "partial": idx < len(pairs),
                    },
                }, f, indent=2)

    print("done", args.out, flush=True)


def analyze(args):
    with open(rooted(args.out), "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = data["pairs"]
    cfg = data["config"]
    alphas = cfg["alphas"]
    names = cfg.get("evaluated", [p for p in cfg["profiles"] if p != "Neutral Baseline"])
    traits = ["directness", "formality", "verbosity"]

    summary: Dict[str, Any] = {"n_pairs": len(pairs), "control": data["control"], "profiles": {}}
    print(f"N pairs = {len(pairs)}")
    print("control logP:", {k: round(v, 4) for k, v in data["control"].items()})
    header = f"{'profile':18s} {'n':>4s} {'dLogP':>8s} {'improved':>9s} {'agree':>7s} {'p':>9s} {'ctrl':>8s}"
    print(header)
    for name in names:
        deltas = []
        per_trait: Dict[str, List[float]] = {t: [] for t in traits}
        agree = 0
        agree_n = 0
        for row in pairs:
            base = row["scores"].get("baseline")
            steered = row["scores"].get(name)
            if base is None or steered is None:
                continue
            d = steered - base
            deltas.append(d)
            per_trait[row["trait_id"]].append(d)
            a = alphas[name].get(row["trait_id"], 0.0)
            if abs(a) > 1e-6:
                agree_n += 1
                if (d > 0) == (a > 0):
                    agree += 1
        n = len(deltas)
        mean_d = sum(deltas) / n if n else float("nan")
        improved = sum(1 for d in deltas if d > 0) / n if n else 0.0
        p = sign_test_p(agree, agree_n)
        ctrl = data["control"].get(name, float("nan")) - data["control"].get("baseline", float("nan"))
        print(f"{name:18s} {n:4d} {mean_d:8.4f} {improved:9.3f} {agree/agree_n if agree_n else 0:7.3f} {p:9.5f} {ctrl:8.4f}")
        summary["profiles"][name] = {
            "n": n,
            "mean_delta_logp": round(mean_d, 5),
            "fraction_improved": round(improved, 4),
            "direction_agreement": round(agree / agree_n if agree_n else 0.0, 4),
            "agreement_n": agree_n,
            "sign_test_p": p,
            "control_delta": round(ctrl, 5),
            "per_trait": {t: round(sum(v) / len(v), 5) if v else None for t, v in per_trait.items()},
            "per_trait_n": {t: len(v) for t, v in per_trait.items()},
            "alphas": alphas[name],
        }
        print("   per-trait:", {t: (round(sum(v)/len(v), 4) if v else None) for t, v in per_trait.items()})

    with open(rooted("artifacts/datasets/eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote artifacts/datasets/eval_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pairs", default="artifacts/datasets/eval_pairs_300.json")
    ap.add_argument("--out", default="artifacts/datasets/eval_results.json")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze(a)
    else:
        run(a)
