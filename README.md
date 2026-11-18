# MBCT — Model-Independent Personality Steering

> **Contrastive Activation Steering** — inject personality trait vectors into a frozen transformer's residual stream at inference time. No fine-tuning. No weight modification.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)](https://fastapi.tiangolo.com)
[![Phi-3 Mini](https://img.shields.io/badge/model-Phi--3--Mini--4k-purple)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

---

## What This Is

MBCT (Model-Based Contrastive Traits) is a **personality steering engine** that:

1. **Calibrates** once — generates contrastive response pairs via Ollama (phi3), runs teacher-forced forward passes through Phi-3 Mini, computes mean activation differences per layer, and stores unit-normalised steering vectors in a **Model Calibration Profile (MCP)**
2. **Steers at inference time** — compiles a **Universal Behaviour Profile (UBP, slider 0–100%)** into per-layer alpha weights and injects `h'ₗ = hₗ + Σ α[t,l] · v[t,l]` via PyTorch forward hooks on the frozen model
3. **Never modifies weights** — all intervention is strictly at inference time

```
UBP sliders (0-100%) ──▶ PersonalityCompiler ──▶ CompiledSteeringPlan
                                                        │
Frozen Phi-3 Mini ◀─── forward hooks ◀─── SteeringController
(residual layers 12, 14, 16)                    h' = h + α·v
```

---

## Architecture

| Component | Role |
|---|---|
| **UBP** (Universal Behaviour Profile) | Model-agnostic. Human-readable sliders. Shared across model variants. |
| **MCP** (Model Calibration Profile) | Model-specific. Stores unit-normalised steering vectors per trait per layer. Generated once. |
| **PersonalityCompiler** | Maps UBP → CompiledSteeringPlan with Phase 4 pairwise cosine-similarity check + Gram-Schmidt orthogonalization |
| **SteeringController** | Applies plan via `ActivationHookManager`. Master kill-switch. Streaming + blocking generation. |
| **CalibrationService** | Full pipeline: Ollama pairs → dataset validation → teacher-forced extraction → vector derivation → MCP persistence |

---

## Traits

| Trait | Range | Effect |
|---|---|---|
| `directness` | 0–100% | 100% = blunt, no hedging. 0% = circumspect, deferential |
| `formality` | 0–100% | 100% = professional register. 0% = casual conversation |
| `verbosity` | 0–100% | 100% = detailed, expansive. 0% = terse, compressed |
| `empathy` | 0–100% | UBP-only (future calibration) |
| `initiative` | 0–100% | UBP-only (future calibration) |

---

## Any Local Model Support

MBCT is architecture-agnostic. You can select, load, and steer **any local model**:

1. **Auto-Discovered Models**: Scans `~/.cache/huggingface/hub/` for any locally cached models (e.g. `microsoft/Phi-3-mini-4k-instruct`, `Qwen/Qwen2.5-0.5B-Instruct`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`), reports weight files and download status.
2. **Local Ollama Models**: Auto-discovers installed Ollama models (`phi3`, `qwen2.5-coder:7b`, etc.) to use as contrastive pair calibration providers.
3. **Custom Local Folders or HF Repo IDs**: Pass any checkpoint directory (e.g. `C:\models\checkpoint`) or repo name in the UI or via `POST /api/models/load`.
4. **Dynamic Transformer Introspection**:
   - `ActivationHookManager` automatically resolves transformer decoder layer blocks across architectures (`model.layers`, `transformer.h`, `model.decoder.layers`, `gpt_neox.layers`, `transformer.layers`, or recursive `nn.ModuleList` inspection).
   - Hidden sizes and layer counts are dynamically extracted from model configurations.
   - Target layers default to optimal semantic depths (35%–65% depth) if not manually specified.
5. **Model-Aware Profiles**: Each model persists its own Model Calibration Profile (e.g. `artifacts/profiles/mcp_{clean_model_key}.json`). Switching models auto-swaps and attaches the matching calibration.

---

## Presets

| Preset | Traits |
|---|---|
| 🏗️ Direct Architect | directness=95%, formality=80%, verbosity=30% |
| 🎓 Friendly Mentor | directness=30%, formality=20%, verbosity=75%, empathy=90% |
| 🔍 Brutal Reviewer | directness=100%, formality=50%, verbosity=25% |
| ⚖️ Neutral Baseline | all=50% (zero steering effect) |

---

## Setup

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai) with `phi3` model pulled (`ollama pull phi3`)
- Phi-3-mini-4k-instruct model cached locally via HuggingFace

### Install

```bash
cd backend
pip install -r requirements.txt
```

### Run

```bash
# Terminal 1 — start Ollama (calibration provider)
ollama serve

# Terminal 2 — start MBCT server
cd MBCT
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8088
```

Then open **http://127.0.0.1:8088** — the Personality Studio UI loads.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | System status, model loaded, calibration state |
| `GET` | `/api/models` | Supported presets, discovered local HF models, and Ollama models |
| `POST` | `/api/models/load` | Dynamically load ANY local model (preset, HF ID, or folder path) |
| `POST` | `/api/models/scan` | Re-scan local HuggingFace cache and Ollama models |
| `GET` | `/api/traits` | Available traits from prompt registry |
| `GET` | `/api/profiles/behaviour` | Current UBP |
| `PUT` | `/api/profiles/behaviour` | Update UBP (recompiles + applies plan immediately) |
| `GET` | `/api/profiles/presets` | 4 preset configurations |
| `POST` | `/api/steering/generate` | Blocking generation (steered + baseline) |
| `POST` | `/api/steering/generate/stream` | **SSE streaming** — tokens via `TextIteratorStreamer` |
| `POST` | `/api/steering/toggle` | Master kill-switch on/off |
| `GET` | `/api/calibrations/current` | Current MCP report |
| `POST` | `/api/calibrations/run` | Launch async calibration job → returns `job_id` |
| `GET` | `/api/calibrations/status/{job_id}` | Poll calibration progress |
| `GET` | `/api/datasets/{trait}` | Stored contrastive dataset pairs |
| `POST` | `/api/providers/test` | Test Ollama connection |

---

## Phases

- ✅ **Phase 0** — Feasibility spike: hook residual stream, verify measurable output shift
- ✅ **Phase 1** — Single-trait (directness) contrastive dataset + vector extraction
- ✅ **Phase 2** — Multi-trait calibration engine (directness, formality, verbosity) + MCP stored to disk
- ✅ **Phase 3** — Full product UI + live inference playground + SSE streaming + async calibration
- ✅ **Phase 4** — Trait composition bounds: pairwise cosine-similarity check + Gram-Schmidt orthogonalization
- 🔄 **Phase 5** — 2nd model support (Qwen2.5-0.5B)
- 🔄 **Phase 6** — External memory / context store

---

## Project Structure

```
MBCT/
├── artifacts/
│   ├── datasets/
│   │   ├── calibration_directness.json
│   │   ├── calibration_formality.json
│   │   └── calibration_verbosity.json
│   └── profiles/
│       └── mcp_phi3.json          ← Calibrated steering vectors
├── backend/
│   └── app/
│       ├── core/types.py           ← Pydantic domain models (UBP, MCP, CompiledSteeringPlan)
│       ├── prompt_registry/registry.py  ← Trait definitions + contrastive system prompts
│       ├── providers/ollama_provider.py ← Ollama HTTP adapter
│       ├── dataset/
│       │   ├── generator.py        ← Contrastive pair generation with gold seeds
│       │   └── validator.py        ← Schema, length-confound, trait-isolation validation
│       ├── models/registry.py      ← ModelLoader (Phi-3 Mini, local_files_only)
│       ├── steering/
│       │   ├── hooks.py            ← ActivationHookManager (forward hooks, kill-switch)
│       │   ├── deriver.py          ← VectorDeriver (mean diff, unit normalisation)
│       │   ├── controller.py       ← SteeringController (generate + generate_stream)
│       │   └── calibrator.py       ← CalibrationService (full pipeline)
│       ├── profiles/compiler.py    ← PersonalityCompiler (Phase 4 orthogonalization)
│       ├── static/index.html       ← 5-tab Personality Studio UI
│       └── main.py                 ← FastAPI application (14 endpoints)
├── calibrate_initial_mcp.py        ← Standalone calibration runner
├── test_spike.py                   ← Phase 0/1 feasibility spike
├── .gitignore
└── README.md
```

---

## Key Design Decisions

- **No weight modification** — all intervention via forward hooks on frozen model
- **UBP/MCP separation** — user sliders are model-agnostic; vectors are model-specific
- **Orthogonalization** — prevents trait interference when multiple sliders are non-neutral (Phase 4)
- **Async calibration** — Ollama pair generation + activation extraction can take minutes; job system prevents UI blocking
- **SSE streaming** — `TextIteratorStreamer` runs in a daemon thread; tokens pushed to asyncio queue and yielded as Server-Sent Events
- **local_files_only=True** — all model loading is offline; no HuggingFace network calls

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for the full text.

You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this software, provided the copyright notice and permission notice are included in all copies or substantial portions.
