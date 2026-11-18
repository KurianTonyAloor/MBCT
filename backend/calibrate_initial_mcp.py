"""Script to calibrate Directness, Formality, and Verbosity on Phi-3 Mini and store the MCP."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.steering.calibrator import CalibrationService
from app.core.types import QualityPreset


def run():
    print("=== Running MVP Calibration for Phi-3 Mini ===", flush=True)
    service = CalibrationService(artifacts_dir="artifacts")
    mcp = service.calibrate_traits(
        model_key="phi-3-mini",
        traits=["directness", "formality", "verbosity"],
        target_layers=[12, 14, 16],
        preset=QualityPreset.ECONOMICAL
    )
    print("\nCalibration Complete!", flush=True)
    print(f"MCP ID: {mcp.mcp_id}", flush=True)
    print(f"Calibrated Traits: {mcp.traits}", flush=True)
    print(f"Report: {mcp.evaluation_report}", flush=True)


if __name__ == "__main__":
    run()
