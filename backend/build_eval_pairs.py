"""Build a held-out evaluation set of 300 contrastive pairs (100 per calibrated trait).

Uses the project's trait system prompts + DatasetValidator, over scenarios that never
appear in the calibration seeds or DEFAULT_SCENARIOS, so the pairs stay held out from the
MCP vectors. Checkpoints every accepted pair, so the build is resumable.

Output: artifacts/datasets/eval_pairs_300.json
"""

import os
import sys
import json
import time
import urllib.request
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.dataset.validator import DatasetValidator
from app.prompt_registry.registry import get_trait

TARGET_PER_TRAIT = 100
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "artifacts", "datasets", "eval_pairs_300.json")
ENDPOINT = "http://127.0.0.1:11434"
MODEL = "phi3"

TOPICS: List[str] = [
    "Kubernetes", "microservices", "PostgreSQL", "Redis", "Kafka", "GraphQL", "REST endpoints",
    "database indexes", "CI pipelines", "blue-green deploys", "feature flags", "vector databases",
    "serverless functions", "message queues", "caching layers", "load balancers", "table sharding",
    "read replicas", "OAuth2", "JWTs", "TLS certificates", "rate limiting", "structured logging",
    "distributed tracing", "unit testing", "integration tests", "monorepos", "trunk-based development",
    "code review", "inline documentation", "on-call rotations", "incident postmortems",
    "capacity planning", "cloud cost optimisation", "data warehousing", "ETL pipelines",
    "stream processing", "event sourcing", "CQRS", "the saga pattern", "service mesh",
    "API gateways", "schema migrations", "connection pooling", "Python asyncio", "Go goroutines",
    "Rust ownership", "TypeScript generics", "SQL window functions", "idempotency keys",
]

TEMPLATES: Dict[str, List[str]] = {
    "directness": [
        "Assess whether {t} fits a three-person engineering team with no dedicated ops staff",
        "Give an unhedged verdict on adopting {t} for a hobby project that may grow",
    ],
    "formality": [
        "Write a client-facing note explaining our current position on {t}",
        "Brief a new contractor on how we currently handle {t}",
        "Summarise for a vendor what our team decided about {t}",
    ],
    "verbosity": [
        "Explain how {t} works to a mid-level developer",
        "Describe the main trade-offs of {t} to someone choosing an approach",
    ],
}


def scenarios_for(trait_id: str) -> List[str]:
    return [tpl.format(t=topic) for tpl in TEMPLATES[trait_id] for topic in TOPICS]


# Token budget per trait: the JSON shell plus the two responses. Directness/formality
# pairs are length-balanced on purpose (a length confound is what corrupts the directness
# vector); verbosity pairs need a real length gap, so they cost more tokens.
WORD_TARGETS = {
    "directness": ("Each response must be 12-18 words.", 190),
    "formality": ("Each response must be 12-18 words.", 190),
    "verbosity": (
        "target_response must be 35-45 words and contrast_response must be 8-12 words.", 300
    ),
}


def call_ollama(prompt: str, system_prompt: str, num_predict: int = 190) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.6, "top_p": 0.9, "num_predict": num_predict, "num_ctx": 2048},
    }
    req = urllib.request.Request(
        f"{ENDPOINT}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())["response"]


def parse_pair(raw: str, scenario: str) -> Dict[str, Any]:
    try:
        rec = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in provider output")
        rec = json.loads(raw[start:end + 1])
    rec.setdefault("scenario", scenario)
    return rec


def text_field(rec: Dict[str, Any], key: str) -> str:
    value = rec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"field '{key}' is not a non-empty string")
    return value.strip().strip('"')


def load_existing() -> List[Dict[str, Any]]:
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save(pairs: List[Dict[str, Any]]):
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
    os.replace(tmp, OUT_PATH)


def main():
    pairs = load_existing()
    counts = {t: sum(1 for p in pairs if p["trait_id"] == t) for t in TEMPLATES}
    print(f"resuming with {len(pairs)} pairs: {counts}", flush=True)

    for trait_id in ("directness", "formality", "verbosity"):
        system_prompt = get_trait(trait_id).system_prompt
        queue = scenarios_for(trait_id)
        seen = {p["scenario"] for p in pairs if p["trait_id"] == trait_id}
        t0 = time.time()
        for scenario in queue:
            if counts[trait_id] >= TARGET_PER_TRAIT:
                break
            if scenario in seen:
                continue
            word_rule, budget = WORD_TARGETS[trait_id]
            instruction = (
                f"Generate 1 contrastive pair for scenario: '{scenario}'.\n"
                f"{word_rule} Return ONLY JSON:\n"
                '{"scenario": "...", "user_prompt": "...", "target_response": "...", '
                '"contrast_response": "...", "invariant_summary": "..."}'
            )
            try:
                raw = call_ollama(instruction, system_prompt, num_predict=budget)
                rec = parse_pair(raw, scenario)
            except Exception as exc:
                print(f"[{trait_id}] skip ({exc})", flush=True)
                continue
            ok, val = DatasetValidator.validate_example(rec, trait_id)
            if not ok:
                print(f"[{trait_id}] reject {val.notes}", flush=True)
                continue
            try:
                fields = {
                    "user_prompt": text_field(rec, "user_prompt"),
                    "positive_response": text_field(rec, "target_response"),
                    "contrast_response": text_field(rec, "contrast_response"),
                }
                invariant = rec.get("invariant_summary")
                invariant = invariant.strip() if isinstance(invariant, str) else ""
            except ValueError as exc:
                print(f"[{trait_id}] reject malformed record ({exc})", flush=True)
                continue
            counts[trait_id] += 1
            pairs.append({
                "example_id": f"eval-{trait_id}-{counts[trait_id]:03d}",
                "trait_id": trait_id,
                "domain": "held_out",
                "scenario": scenario,
                "user_prompt": fields["user_prompt"],
                "positive_response": fields["positive_response"],
                "contrast_response": fields["contrast_response"],
                "invariant_summary": invariant,
                "length_ratio": val.length_confound_ratio,
            })
            seen.add(scenario)
            save(pairs)
            if counts[trait_id] % 10 == 0:
                rate = (time.time() - t0) / max(1, counts[trait_id] - (counts[trait_id] - 10))
                print(f"[{trait_id}] {counts[trait_id]}/{TARGET_PER_TRAIT} "
                      f"({len(pairs)} total, {rate:.1f}s/pair)", flush=True)
        print(f"[{trait_id}] finished at {counts[trait_id]}", flush=True)

    print("DONE", len(pairs), OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
