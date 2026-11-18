"""Dataset generation engine that generates and validates contrastive calibration pairs."""

import uuid
from typing import List, Dict, Any, Optional
from app.core.types import CalibrationExample, QualityPreset
from app.prompt_registry.registry import get_trait
from app.providers.ollama_provider import OllamaProvider
from app.dataset.validator import DatasetValidator


DEFAULT_SCENARIOS = {
    "directness": [
        "Review a proposed database schema for a multi-tenant SaaS application",
        "Advise whether to refactor a monolithic codebase before a product launch",
        "Assess whether to adopt microservices for a 3-person startup team",
        "Give feedback on a poorly performing SQL query with Cartesian joins",
        "Evaluate whether a candidate's resume meets senior engineering criteria"
    ],
    "formality": [
        "Draft a message to a client explaining a scheduled maintenance downtime",
        "Respond to a coworker asking how to set up their local dev environment",
        "Explain to a team lead why a sprint deadline might slip by two days",
        "Welcome a new contributor to an open-source project repository",
        "Summarize the findings of a quarterly performance retro"
    ],
    "verbosity": [
        "Explain how python generators and the yield keyword work",
        "Explain what an HTTP 429 status code means and how to handle it",
        "Provide steps to resolve a git merge conflict in a rebase",
        "Explain the difference between process and thread in operating systems",
        "Describe how asymmetric encryption (public/private keys) works"
    ]
}


# Curated Gold-Standard Seed dataset for immediate Phase 0/1 testing & reproducibility
CURATED_GOLD_SEEDS: Dict[str, List[Dict[str, Any]]] = {
    "directness": [
        {
            "scenario": "Review a proposed database schema",
            "user_prompt": "Would this schema work for a many-to-many relationship without a junction table?",
            "target_response": "No. A relational database requires a junction table with foreign keys referencing both entities to model a many-to-many relationship reliably.",
            "contrast_response": "I think there might be some structural limitations with that approach. Generally speaking, in relational design, it's often advisable to look into creating an intermediate table connecting both entities.",
            "invariant_summary": "Both responses identify that a junction table is necessary for a many-to-many relationship."
        },
        {
            "scenario": "Advise on microservices for a 3-person team",
            "user_prompt": "Should our 3-person engineering team split our app into microservices now?",
            "target_response": "No. Microservices introduce distributed complexity, network overhead, and deployment friction that will slow down a three-person team. Stay on a modular monolith.",
            "contrast_response": "That's an interesting question with multiple trade-offs. While microservices offer decoupled scaling and service autonomy, you might also find that the operational overhead could potentially impact your current delivery velocity.",
            "invariant_summary": "Both responses caution against moving to microservices due to operational overhead on a small team."
        },
        {
            "scenario": "Evaluate adopting Kubernetes for a static landing page",
            "user_prompt": "Should we deploy our static landing page on a Kubernetes cluster?",
            "target_response": "No. Deploying a static site on Kubernetes is wasteful and unnecessarily complex. Use a CDN or static hosting service like S3/Cloudflare.",
            "contrast_response": "While Kubernetes is certainly very capable and flexible, you might want to weigh whether the management overhead is justified when simpler options like static object storage or CDN hosting are readily available.",
            "invariant_summary": "Both conclude Kubernetes is over-engineered for a static page and recommend static hosting."
        }
    ],
    "formality": [
        {
            "scenario": "Explain scheduled maintenance downtime",
            "user_prompt": "How should we notify customers about 20 minutes of downtime tomorrow night?",
            "target_response": "Please be advised that scheduled infrastructure maintenance will take place tomorrow at 22:00 UTC. The platform will be temporarily inaccessible for approximately twenty minutes. We appreciate your patience.",
            "contrast_response": "Hey everyone! Quick heads up that we're doing some maintenance tomorrow night around 10 PM UTC. Things will be down for about 20 mins while we update things. Thanks for hanging tight!",
            "invariant_summary": "Both inform customers of 20 minutes of maintenance downtime at 22:00 UTC."
        },
        {
            "scenario": "Explain sprint delay to team lead",
            "user_prompt": "How do I tell my lead the auth task is running 1 day late?",
            "target_response": "I am writing to provide an update regarding the authentication module. Due to unforeseen edge cases in session revocation, completion will be delayed by one business day. I anticipate delivery by Wednesday afternoon.",
            "contrast_response": "Hey, just wanted to give you a quick ping on the auth task. Ran into a couple tricky edge cases with session handling, so looks like I'll need another day on this. Should have it wrapped up by Wednesday afternoon!",
            "invariant_summary": "Both communicate a 1-day delay on the auth module due to edge cases, with completion on Wednesday."
        }
    ],
    "verbosity": [
        {
            "scenario": "Explain HTTP 429 status code",
            "user_prompt": "What does an HTTP 429 status code mean?",
            "target_response": "An HTTP 429 'Too Many Requests' response indicates that the client has sent too many requests in a given amount of time, exceeding the rate limits set by the server. Servers frequently include a 'Retry-After' header indicating how long to wait before sending another request.",
            "contrast_response": "HTTP 429 means 'Too Many Requests'—you hit the server's rate limit. Check the Retry-After header and back off.",
            "invariant_summary": "Both state that HTTP 429 means Too Many Requests and points to rate limits and Retry-After."
        },
        {
            "scenario": "Explain git rebase conflict resolution",
            "user_prompt": "How do I fix a merge conflict during a git rebase?",
            "target_response": "During a git rebase conflict, git pauses at the conflicting commit. You must inspect the conflicted files, resolve the conflict markers (<<<<<<<, =======, >>>>>>>), stage the resolved files with 'git add <file>', and then execute 'git rebase --continue' to apply subsequent commits.",
            "contrast_response": "Edit the conflicted files to remove the conflict markers, run `git add <file>`, then run `git --continue`.",
            "invariant_summary": "Both instruct to manually fix conflict markers, run git add, and git rebase --continue."
        }
    ]
}


class DatasetGenerator:
    def __init__(self, provider: Optional[OllamaProvider] = None):
        self.provider = provider or OllamaProvider()

    def generate_for_trait(
        self,
        trait_id: str,
        quality_preset: QualityPreset = QualityPreset.BALANCED,
        use_gold_seeds: bool = True
    ) -> List[CalibrationExample]:
        """Generate validated contrastive pairs for a specific trait."""
        trait = get_trait(trait_id)
        results: List[CalibrationExample] = []

        # 1. First include gold standard seeds if available
        if use_gold_seeds and trait_id in CURATED_GOLD_SEEDS:
            for i, seed in enumerate(CURATED_GOLD_SEEDS[trait_id]):
                is_valid, val_res = DatasetValidator.validate_example(seed, trait_id)
                results.append(
                    CalibrationExample(
                        example_id=f"{trait_id}-gold-{i+1:03d}",
                        trait_id=trait_id,
                        intensity=0.8,
                        domain="curated",
                        scenario=seed["scenario"],
                        user_prompt=seed["user_prompt"],
                        positive_response=seed["target_response"],
                        contrast_response=seed["contrast_response"],
                        invariant_summary=seed["invariant_summary"],
                        validation=val_res
                    )
                )

        # 2. If needed, generate additional scenarios via Ollama
        target_count = 3 if quality_preset == QualityPreset.ECONOMICAL else (6 if quality_preset == QualityPreset.BALANCED else 10)
        needed = target_count - len(results)

        if needed > 0 and self.provider:
            scenarios = DEFAULT_SCENARIOS.get(trait_id, ["General technical query", "General advice query"])
            for idx, scenario in enumerate(scenarios[:needed]):
                try:
                    raw_record = self.provider.generate_contrastive_pair(
                        trait_id=trait_id,
                        scenario=scenario,
                        system_prompt=trait.system_prompt
                    )
                    is_valid, val_res = DatasetValidator.validate_example(raw_record, trait_id)
                    if is_valid:
                        results.append(
                            CalibrationExample(
                                example_id=f"{trait_id}-gen-{len(results)+1:03d}",
                                trait_id=trait_id,
                                intensity=0.8,
                                domain="generated",
                                scenario=raw_record.get("scenario", scenario),
                                user_prompt=raw_record["user_prompt"],
                                positive_response=raw_record["target_response"],
                                contrast_response=raw_record["contrast_response"],
                                invariant_summary=raw_record["invariant_summary"],
                                validation=val_res
                            )
                        )
                except Exception as e:
                    print(f"Warning: generation for '{scenario}' failed: {e}")

        return results
