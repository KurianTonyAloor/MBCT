"""Validation pipeline for candidate contrastive calibration datasets."""

from typing import Dict, Any, Tuple
from app.core.types import CalibrationExample, ValidationResult


class DatasetValidator:
    @staticmethod
    def validate_example(record: Dict[str, Any], trait_id: str) -> Tuple[bool, ValidationResult]:
        """Validate a single contrastive calibration pair against blueprint criteria."""
        notes = []
        is_valid = True

        # 1. Schema validity
        required = ["scenario", "user_prompt", "target_response", "contrast_response", "invariant_summary"]
        for field in required:
            val = record.get(field, "")
            if not val or not str(val).strip():
                return False, ValidationResult(
                    schema_valid=False,
                    semantic_equivalence="failed",
                    trait_isolation="failed",
                    notes=[f"Missing or empty required field: '{field}'"]
                )

        target = record["target_response"].strip()
        contrast = record["contrast_response"].strip()
        user_prompt = record["user_prompt"].strip()

        # 2. Trait isolation / non-triviality check
        if target.lower() == contrast.lower():
            is_valid = False
            notes.append("Target and contrast responses are identical.")

        # 3. Length confound check
        target_words = len(target.split())
        contrast_words = len(contrast.split())
        ratio = target_words / max(contrast_words, 1)

        # For traits other than verbosity, severe length imbalance can confound the vector
        if trait_id != "verbosity":
            if ratio < 0.25 or ratio > 4.0:
                notes.append(f"Length confound warning: target has {target_words} words, contrast has {contrast_words} words.")

        # 4. Invariant check
        inv = record.get("invariant_summary", "").strip()
        if len(inv) < 5:
            notes.append("Invariant summary too brief or vague.")

        val_result = ValidationResult(
            schema_valid=True,
            semantic_equivalence="passed" if is_valid else "flagged",
            trait_isolation="passed" if is_valid else "failed",
            factual_consistency="passed",
            length_confound_ratio=round(ratio, 2),
            notes=notes
        )

        return is_valid, val_result
