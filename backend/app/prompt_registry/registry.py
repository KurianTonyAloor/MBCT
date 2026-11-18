"""Prompt registry storing curated, versioned prompts for contrastive trait calibration."""

from typing import Dict, Any, List
from pydantic import BaseModel


class TraitDefinition(BaseModel):
    trait_id: str
    display_name: str
    definition: str
    target_pole: str
    contrast_pole: str
    potential_confound: str
    domains: List[str]
    system_prompt: str


TRAIT_CATALOGUE: Dict[str, TraitDefinition] = {
    "directness": TraitDefinition(
        trait_id="directness",
        display_name="Directness",
        definition="How explicitly and unambiguously the model states conclusions, answers questions, and delivers assessments.",
        target_pole="High directness: Direct, concise, unambiguous, decisive assessment upfront, minimal preamble.",
        contrast_pole="Low directness: Qualified, indirect, hedged language, softened phrasing, polite conversational filler.",
        potential_confound="Confidence, politeness, assertiveness.",
        domains=["technical", "professional", "planning", "everyday_help", "decision_making"],
        system_prompt="""You are generating controlled behavioural calibration data for an LLM activation-steering system.
OBJECTIVE:
Create contrastive pairs for exactly one target behaviour: DIRECTNESS.

TARGET POLE: High directness (unambiguous, immediate answer, concise rationale).
CONTRAST POLE: Low directness (indirect, hedged, softened phrasing, polite preamble).

REQUIREMENTS:
1. Each pair MUST have the exact same user prompt.
2. Both target and contrast responses MUST give the exact same underlying answer, recommendation, and facts.
3. Only the style/directness varies. Do NOT make the contrast response less technically accurate or fail the task.
4. Keep the token lengths reasonably balanced to avoid length confounds.
5. Return JSON matching the requested schema."""
    ),
    "formality": TraitDefinition(
        trait_id="formality",
        display_name="Formality",
        definition="Degree of formal language, professional register, grammatical precision, and tone.",
        target_pole="High formality: Professional register, precise vocabulary, polished corporate/academic tone, no colloquialisms.",
        contrast_pole="Low formality: Casual, conversational, relaxed language, friendly colloquial tone, contractions.",
        potential_confound="Vocabulary sophistication, sentence complexity.",
        domains=["business", "coding", "email_drafting", "general_advice", "academic"],
        system_prompt="""You are generating controlled behavioural calibration data for an LLM activation-steering system.
OBJECTIVE:
Create contrastive pairs for exactly one target behaviour: FORMALITY.

TARGET POLE: High formality (professional, formal register, meticulous tone, structured).
CONTRAST POLE: Low formality (casual, warm, colloquial, conversational register).

REQUIREMENTS:
1. Each pair MUST have the exact same user prompt.
2. Both target and contrast responses MUST give the exact same technical conclusion and recommendation.
3. Only the formality/register varies.
4. Keep lengths reasonably balanced.
5. Return JSON matching the requested schema."""
    ),
    "verbosity": TraitDefinition(
        trait_id="verbosity",
        display_name="Verbosity",
        definition="Amount of detailed explanation, context, and elaboration supplied.",
        target_pole="High verbosity: Comprehensive, in-depth elaboration, additional context, thorough rationale.",
        contrast_pole="Low verbosity: Highly concise, bulleted or short sentences, essential information only.",
        potential_confound="Task completeness, reasoning depth.",
        domains=["troubleshooting", "concept_explanation", "code_review", "recommendations", "planning"],
        system_prompt="""You are generating controlled behavioural calibration data for an LLM activation-steering system.
OBJECTIVE:
Create contrastive pairs for exactly one target behaviour: VERBOSITY.

TARGET POLE: High verbosity (detailed elaboration, thorough context and breakdown).
CONTRAST POLE: Low verbosity (concise, succinct, zero unnecessary words, punchy).

REQUIREMENTS:
1. Each pair MUST have the exact same user prompt.
2. Both target and contrast responses MUST provide accurate and sufficient answers.
3. Return JSON matching the requested schema."""
    )
}


def get_trait(trait_id: str) -> TraitDefinition:
    if trait_id not in TRAIT_CATALOGUE:
        raise ValueError(f"Unknown trait '{trait_id}'. Available: {list(TRAIT_CATALOGUE.keys())}")
    return TRAIT_CATALOGUE[trait_id]


def list_traits() -> List[Dict[str, Any]]:
    return [
        {
            "trait_id": t.trait_id,
            "display_name": t.display_name,
            "definition": t.definition,
            "potential_confound": t.potential_confound,
            "domains": t.domains
        }
        for t in TRAIT_CATALOGUE.values()
    ]
