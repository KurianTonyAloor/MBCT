# MBCT — Model-Independent Personality Steering

> **Personality without fine-tuning** — take any local model you already have on disk, and steer its personality at inference time by injecting small activation vectors into its residual stream. No retraining. No weight changes. Runs on the same hardware the model already runs on.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)](https://fastapi.tiangolo.com)
[![Phi-3 Mini](https://img.shields.io/badge/model-Phi--3--Mini--4k-purple)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

---

## What This Is

MBCT is a way to change how a model speaks and acts — its personality — without doing any fine-tuning.

Fine-tuning a model to a specific personality is tedious, resource-heavy, and usually tied to one model snapshot. MBCT takes a different approach: the model stays frozen, and personality is applied at inference time by injecting small steering signals into the model's residual stream. The model's weights are never touched.

That matters for two reasons:

- **Anything can be used on any edge system.** You don't need a training run, a big GPU cluster, or a curated dataset to reshape a model's behaviour. If you already have a local model sitting on disk, MBCT can load it, calibrate personality vectors once for that model, and then let you dial traits up and down at generation time.
- **Personality becomes a live, adjustable thing instead of a model build.** The same model can be switched between direct and cautious, formal and casual, terse and chatty, without rebuilding it. Those adjustments are applied during generation, not baked into weights.

MBCT is built around that idea:

1. **Calibrate once per model** — generate contrastive response pairs (for example through a local Ollama provider), run teacher-forced forward passes through the model, compute the mean activation difference per layer, and store the unit-normalised steering vectors in a **Model Calibration Profile (MCP)**
2. **Steer at inference time** — compile a **Universal Behaviour Profile (UBP, slider 0–100%)** into per-layer alpha weights and inject `h'ₗ = hₗ + Σ α[t,l] · v[t,l]` via PyTorch forward hooks on the frozen model
3. **Never modify weights** — all intervention is strictly at inference time

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

## Why This Exists

Right now, if you want a model to behave in a particular personality, the usual route is fine-tuning. That is tedious, resource-heavy, and locked to a specific model and snapshot. MBCT is an attempt to make personality customization usable on any edge system, with whatever local model you already have, without retraining.

The goal is practical, not research-demo scale:

- **No fine-tuning.** Personality is applied at inference time through forward hooks on a frozen model.
- **Any local model.** MBCT discovers and loads models from the HuggingFace cache, local Ollama installs, or any custom checkpoint folder you point it at.
- **Edge-relevant.** Calibration only needs to happen once per model. After that, changing personality is a matter of adjusting sliders, not rebuilding anything.
- **Live behaviour, not a new model build.** The same model can be pushed into different behavioural modes during generation.

In short: MBCT is what it means for a local model to have an adjustable personality without the cost and friction of fine-tuning.

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
| `POST` | `/api/steering/generate` | Blocking generation (steered + baseline, memory context) |
| `POST` | `/api/steering/generate/stream` | **SSE streaming** — tokens via `TextIteratorStreamer` with memory |
| `POST` | `/api/steering/toggle` | Master kill-switch on/off |
| `GET` | `/api/calibrations/current` | Current MCP report |
| `POST` | `/api/calibrations/run` | Launch async calibration job → returns `job_id` |
| `GET` | `/api/calibrations/status/{job_id}` | Poll calibration progress |
| `POST` | `/api/calibrations/validate-compatibility` | Verify MCP compatibility against active model architecture |
| `GET` | `/api/datasets/{trait}` | Stored contrastive dataset pairs |
| `GET` | `/api/providers/config` | Get active provider configuration (masked key) + presets |
| `POST` | `/api/providers/config` | Update provider endpoint, API key, and model ID |
| `POST` | `/api/providers/test` | Test provider connection with minimal token overhead |
| `GET` | `/api/memory` | List stored memories (filterable by type) |
| `POST` | `/api/memory` | Store new memory item (fact, preference, episodic) |
| `PUT` | `/api/memory/{id}` | Update or toggle memory item |
| `DELETE` | `/api/memory/{id}` | Delete memory item |
| `POST` | `/api/memory/toggle` | Master external memory toggle on/off |
| `POST` | `/api/memory/preview` | Preview memory retrieval for a test query |

---

## Phases

- ✅ **Phase 0** — Feasibility spike: hook residual stream, verify measurable output shift
- ✅ **Phase 1** — Single-trait (directness) contrastive dataset + vector extraction
- ✅ **Phase 2** — Multi-trait calibration engine (directness, formality, verbosity) + MCP stored to disk
- ✅ **Phase 3** — Full product UI + live inference playground + SSE streaming + async calibration
- ✅ **Phase 4** — Trait composition bounds: pairwise cosine-similarity check + Gram-Schmidt orthogonalization
- ✅ **Phase 5** — Architecture-agnostic local model discovery, prompt templates, and multi-model support
- ✅ **Phase 6** — Model revision fingerprinting & 4-tier calibration compatibility validation
- ✅ **Phase 7** — Pluggable calibration providers (Ollama, OpenAI, DeepSeek, Groq, OpenRouter, LM Studio)
- ✅ **Phase 8** — External memory integration with relevance retrieval and structured context injection

---

## Project Structure

```
MBCT/
├── artifacts/
│   ├── datasets/
│   │   ├── calibration_directness.json
│   │   ├── calibration_formality.json
│   │   └── calibration_verbosity.json
│   ├── memory/
│   │   └── store.json             ← External memory persistence
│   ├── profiles/
│   │   └── mcp_phi3.json          ← Calibrated steering vectors
│   └── providers/
│       └── active_config.json     ← Provider settings
├── backend/
│   └── app/
│       ├── core/types.py           ← Pydantic domain models (UBP, MCP, Memory, Providers, Compatibility)
│       ├── prompt_registry/registry.py  ← Trait definitions + contrastive system prompts
│       ├── providers/
│       │   ├── base.py             ← BaseCalibrationProvider ABC
│       │   ├── manager.py          ← ProviderManager (presets, config persistence, key masking)
│       │   ├── ollama_provider.py  ← Ollama HTTP adapter
│       │   └── openai_provider.py  ← OpenAI-compatible provider (Groq, DeepSeek, LM Studio, etc.)
│       ├── dataset/
│       │   ├── generator.py        ← Contrastive pair generation via configured provider
│       │   └── validator.py        ← Schema, length-confound, trait-isolation validation
│       ├── models/registry.py      ← ModelLoader (architecture-aware chat templates, local_files_only)
│       ├── memory/
│       │   └── store.py            ← ExternalMemoryStore (Jaccard + tag relevance, context injection)
│       ├── steering/
│       │   ├── hooks.py            ← ActivationHookManager (forward hooks, multi-layer heuristic scan)
│       │   ├── deriver.py          ← VectorDeriver (mean diff, unit normalisation)
│       │   ├── controller.py       ← SteeringController (generate + stream + memory injection)
│       │   ├── calibrator.py       ← CalibrationService (pipeline with provider manager)
│       │   └── compatibility.py    ← CompatibilityValidator (deterministic SHA256 fingerprinting)
│       ├── profiles/compiler.py    ← PersonalityCompiler (orthogonalization)
│       ├── static/index.html       ← 6-tab Personality Studio UI (Playground, UBP, MCP, Calibrate, Memory, Models)
│       └── main.py                 ← FastAPI application (25 endpoints)
├── calibrate_initial_mcp.py        ← Standalone calibration runner
├── test_spike.py                   ← Phase 0/1 feasibility spike
├── LICENSE                         ← MIT License
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
