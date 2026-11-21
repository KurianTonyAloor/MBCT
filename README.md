# MBCT — Model Based Character Tuning

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

## LLM-as-a-Judge Evaluation

Measuring steering by reading one output is guesswork — a pushed slider and a better answer are not the same thing. So every profile is evaluated as a **steered-vs-baseline pair**: the same model, the same prompt, the same decoding settings, one arm steered and one unsteered. The baseline is the control that separates *"the model happened to answer well"* from *"the steering changed the answer"*.

`POST /api/steering/generate` returns both arms in one response, which is exactly the unit the judge scores.

### Rubric

A judge reads both arms blind to the active weights and scores each 0–5:

| Dimension | Weight | What it measures |
|---|---|---|
| Trait adherence | 65% | Split across the dialled traits (directness 30 / formality 20 / verbosity 15): did each slider move **its own** target, in the intended direction? |
| Baseline contrast | 10% | Is the steered arm visibly different from the control, in the intended direction? |
| Semantic preservation | 15% | Same question answered, no topic drift, no invented facts |
| Coherence / completeness | 10% | Grammatical, on-topic, and terminating on a complete thought |

Contrast is a property of the *pair*, not of either arm, so it is reported but excluded from both totals. The four per-arm dimensions are renormalised to 100%.

### Run 1 — Direct Architect, pre-fix build → **LOSS**

Prompt: *"Should our 3-person startup adopt Kubernetes?"* · weights: directness +2.70, formality +1.80, verbosity −1.20

- Formality moved as dialled — the one clean signal in the pair.
- Directness and verbosity moved **inverted**: the steered arm hedged more and said less per token than the baseline it was meant to outrank.
- The memory context block was prepended to the user turn, so the model opened by reciting it and never front-loaded an answer.
- Both arms hit the token cap mid-sentence, so the verbosity axis could not be read at all.

Verdict: **loss**. The traits were live, but the prompt assembly was spending the answer's budget on context and the result was worse to read than the unsteered control.

### Run 2 — Direct Architect, current build → **WIN (3.36 vs 2.42)**

Same prompt and weights, after memory moved to a system turn, retrieval was stopword/synonym-filtered, the vector norm was clamped, and injection was budgeted to 5 of 32 layers (`L11, L13, L16, L19, L21`):

| Dimension | Steered | Baseline |
|---|---|---|
| Directness (+2.70) | 3.0 | 2.0 |
| Formality (+1.80) | **5.0** | 2.5 |
| Verbosity (−1.20) | 2.0 | 1.5 |
| Baseline contrast | 3.5 | — |
| Semantic preservation | 3.5 | 4.0 |
| Coherence / completeness | 3.0 | 2.5 |
| **Weighted (90% of rubric)** | **3.36** | **2.42** |

- **Formality** is the strongest evidence: the steered arm sits in a professional register ("may not be the most suitable choice at this stage") where the baseline stays conversational ("can help manage and scale your applications efficiently").
- **Directness** moved but at half strength: the steered arm commits to a position, yet still opens by defining Kubernetes and hedges the verdict with "may not be".
- **Verbosity** did not move. At +1.80 formality against −1.20 verbosity, elaboration pressure wins: the steered arm *adds* a subordinate clause instead of trimming.

### Run 3 — Friendly Mentor, current build → **WIN (2.94 vs 2.61)**

Same prompt. Weights: directness −1.20, formality −1.80, verbosity +1.50. The preset also dials `empathy` to 90%, which has no calibrated vector and therefore does nothing.

| Dimension | Steered | Baseline |
|---|---|---|
| Directness (−1.20) | 3.0 | 2.5 |
| Formality (−1.80) | 3.0 | 2.5 |
| Verbosity (+1.50) | 2.5 | 2.0 |
| Baseline contrast | 2.0 | — |
| Semantic preservation | 3.5 | 4.0 |
| Coherence / completeness | 2.5 | 2.0 |
| **Weighted (90% of rubric)** | **2.94** | **2.61** |

- **All three measurable dials moved in the right direction, and all three margins are small.** The concrete evidence is register, not vocabulary: two contractions the baseline never uses ("that's typically used", "you're working with"), a softened modal ("might not be the most suitable"), and a deferential concession opener ("Given your preference for modular monoliths") where Direct Architect opened with a flat claim.
- **Verbosity produced length, not detail.** At +1.50 the arm spends three sentences and restates the user's own preference before offering any advice, and never names an alternative — where Direct Architect, at −1.20, still got as far as "Docker Compose" in comparable space. The dial added subordinate clauses rather than content.
- **`empathy` at 90% changed nothing.** With no vector to inject it is inert, so "Friendly Mentor" as measured is only "casual": nothing in the response is warm, reassuring, or mentor-like. The preset's largest slider is its most decorative one.
- **Memory handling improved at these alphas, at a cost.** At −1.80 the `[PREFERENCE]` tag no longer leaks — recitation traded for paraphrase (Run 6 shows this does not hold at +3.00). The paraphrase then repeats: the preference is restated in the opening clause, again in "prefer a monolithic approach", and again in the concession. Same fact, three times, out of a truncated budget.
- **Truncated again**, this time mid-clause at "…approach, you". Third run in a row, and in all three the cut lands on the payload (the recommendation) rather than the setup.
- **Imprecision:** "Kubernetes … aligns more with microservices architectures" overstates the coupling. Kubernetes orchestrates any containerised workload, monoliths included.

### Run 4 — Friendly Mentor, code-review prompt → **WIN (3.28 vs 2.83), but neither arm reviews the code**

Prompt: *"Review this code: `for i in range(len(arr)): print(arr[i])`"* · same weights, 80-token budget. This is the first pair where the baseline answers the **same** prompt as the steered arm at an equal budget, so the trait attribution is clean rather than confounded by prompt or cap.

| Dimension | Steered | Baseline |
|---|---|---|
| Directness (−1.20) | 3.0 | 3.0 |
| Formality (−1.80) | **4.0** | 2.5 |
| Verbosity (+1.50) | **3.0** | 2.0 |
| Baseline contrast | 3.0 | — |
| Semantic preservation | 4.0 | 4.0 |
| Coherence / completeness | 2.0 | **2.5** |
| **Weighted (90% of rubric)** | **3.28** | **2.83** |

- **Formality (−1.80) is the cleanest signal across all four runs.** Steered opening: *"The code snippet **you've provided** … **It's** a straightforward approach … there are a few considerations and potential improvements **we can discuss**."* Baseline: *"The provided code snippet … Here's a breakdown of the code and some suggestions for improvement."* Two contractions, a second-person possessive and a collaborative "we" against a clipped impersonal control — a real register shift, with the attribution finally unambiguous.
- **Verbosity (+1.50) added words, not findings.** ~47 words of prose before any content, against ~29 in the baseline. The extra ~18 are pure restatement: *"It's a straightforward approach to accessing and displaying the elements of an array"* repeats the sentence immediately above it. The dial moved in its stated direction and produced negative value.
- **Directness (−1.20) is unreadable on this prompt.** *"there are a few considerations and potential improvements we can discuss"* is appropriately circumspect — and identical in effect to the baseline's *"some suggestions for improvement"*. Both defer, neither delivers, so both arms score 3.0.
- **The steered arm is less complete than the control.** It is cut off *inside* the fenced block of the very code it was asked to review; the baseline finishes the block and reaches `### Analysis / **Purpose**:`. Coherence 2.0 vs 2.5 — at +1.50 the elaboration bought truncation.
- **`empathy` at 90% again contributed nothing.** The collaborative tone comes from lowered formality, not from the preset's largest dial.

The structural lesson: the model's default shape is preamble → echo the code → analyse. The steered arm follows it twice over, adding a `### Original Code:` heading **and** a restating sentence where the baseline has only the code block. At 80 tokens that costs the entire budget. **For open-ended explanation prompts, lowered formality is the measurable win; for task-shaped prompts with a hard payload — code review, anything with a required deliverable — positive verbosity is an active liability, because elaboration competes with the answer for a capped budget.** Friendly Mentor asks for verbosity 75% *and* for a finished review, and at this budget it cannot have both.

### Run 5 — Friendly Mentor, explainer prompt → **TIE (2.78 vs 2.72)**

Prompt: *"Explain what a p-value is to a non-statistician."* · same weights, 80-token budget. This is the first pair in which steering produced **no measurable effect**, and it is the most informative run in the set.

| Dimension | Steered | Baseline |
|---|---|---|
| Directness (−1.20) | 3.0 | 3.0 |
| Formality (−1.80) | 3.0 | 2.5 |
| Verbosity (+1.50) | 2.0 | 2.0 |
| Baseline contrast | **1.0** | — |
| Semantic preservation | 3.0 | 3.0 |
| Coherence / completeness | 2.5 | **3.0** |
| **Weighted (90% of rubric)** | **2.78** | **2.72** |

The 0.06 margin is below what this rubric can resolve. Diff the arms and five steered layers, three traits and a 2.70-wide alpha spread amount to this:

| | Steered | Baseline |
|---|---|---|
| Contraction | *"if **they're** likely to be"* | *"if **they are** likely to be"* |
| Synonym | *"a **real** effect"* | *"a **true** effect"* |
| Terminator | ends at *"get 60 heads"* | ends at *"get 60 heads."* |

Everything between those three points is token-identical. The arms **diverge locally and then resynchronise**: the injected delta was large enough to flip two argmax choices and not large enough to move the continuation, so greedy decoding snapped back onto the same path. At these alphas, on a prompt the model is confident about, the steering signal sits **below the sampling resolution** — not absent, since formality did produce its contraction in the correct direction, but invisible to a prose judge.

- **Content defect, shared by both arms.** *"A p-value is a number that helps us understand if the results … are due to chance or if they're likely to be a real effect"* is the canonical misinterpretation. A p-value is the probability of data at least this extreme **if the null hypothesis were true** — not the probability that the result is due to chance, and not the probability that an effect is real. For the stated audience that inversion is the most consequential error available, and it caps both arms at 3.0.
- **The analogy is well chosen and then abandoned.** 60 heads in 100 flips is p ≈ 0.057 two-sided, a genuinely useful teaching moment just past the conventional threshold that the truncated output never reaches.
- **Why a tie is the most valuable entry here.** A judge that awards every steered arm a win is worthless. Runs 1–4 can all be read as flattery; this one cannot. The rubric independently reported a null result, which is the evidence that it is measuring something.

**Instrument limitation.** Prose judging is too coarse for a two-token delta. The sensitive measurement is already available in the harness: count the fraction of decode positions where the steered argmax differs from the baseline's, per prompt, at temperature 0. This pair would score ~2/80 ≈ 2.5% divergence against Direct Architect's large early split — reproducible, judge-free, and expressible as "Friendly Mentor at these alphas perturbs 2–3% of tokens" rather than "the judge could not tell".

### Run 6 — Brutal Reviewer, architecture prompt → **LOSS (2.22 vs 2.94)**

Prompt: *"Should I rewrite our legacy PHP monolith in Go?"* · weights: directness **+3.00**, verbosity **−1.50**, memory injected, 80-token budget. Formality sits at the preset's 50% (alpha 0), so that dial is untestable in this pair.

| Dimension | Steered | Baseline |
|---|---|---|
| Directness (+3.00) | **2.0** | 3.0 |
| Formality (0.00) | 3.0 | 3.0 |
| Verbosity (−1.50) | 1.5 | **2.5** |
| Baseline contrast | 2.0 | — |
| Semantic preservation | **2.5** | 3.5 |
| Coherence / completeness | 2.0 | **2.5** |
| **Weighted (90% of rubric)** | **2.22** | **2.94** |

This is the first outright loss on the current build and the first Brutal Reviewer run. Both measurable dials moved **inverted**, and the memory layer did most of the damage.

- **Directness +3.00 produced the least direct answer in the set.** Asked a yes/no question, the arm answers: *"Rewriting your legacy PHP monolith in Go **can offer** performance improvements … **However, consider the following** before making the decision:"* — a deferral, not a verdict, followed by a checklist item that hedges twice (*"a rewrite **can be** beneficial"*). The unsteered baseline is **more decisive**: it names the mechanism (*"Go is known for its performance and concurrency model"*), ties it to a testable condition, and commits to *"Go might be a good choice"*.
- **Directness has now inverted in both high-alpha runs** (+2.70 in Run 1, +3.00 here). That is a pattern rather than noise. The working hypothesis: the vector encodes *"restate the premise plainly"* rather than *"answer bluntly"*, which would fit the documented length confound in the directness contrastive pairs — blunt responses were shorter, so part of the direction is recapitulation. On this evidence, high directness is the worst-behaving setting in the app, which is the opposite of what the preset promises.
- **Memory recitation returns at double the earlier severity.** The arm opens with both stored notes verbatim, tags and bullets intact: `• [FACT] Our primary production database is PostgreSQL 16 …` and `• [PREFERENCE] User prefers architectural feedback to prioritize modular monoliths …`. That is ~44 of 81 tokens — **over half the response budget spent quoting the store back at the user.** Run 3's paraphrase fix holds at Friendly Mentor's alphas, not at Brutal Reviewer's.
- **The irrelevant note was retrieved, and the relevant one was ignored.** PostgreSQL 16 on RDS has no bearing on a PHP→Go migration yet consumed ~22 tokens. Meanwhile the `[PREFERENCE]` note names exactly the frame that answers the question — modular monoliths for small teams — and the arm recites it, then says *"a rewrite can be beneficial"* without using it. Run 3 at least wove the preference into the advice; here the cost is paid and the benefit discarded.
- **Verbosity −1.50 inverted as well.** ~49 content words against the baseline's ~59, *while also* spending 44 tokens on memory: the terse dial produced the least information per budget token of any run. It is cut off mid-word at `"Ens"`.
- **Brutal Reviewer as shipped is not brutal.** At 100% directness and 25% verbosity its measurable identity is "a longer, hedged checklist with a memory preamble".

### Cross-run observation: verbosity trades words for payload

Three different behaviours, all failures to deliver detail:

- **Explanation prompts, opposite alphas, same length.** Direct Architect at verbosity **−1.20** produced 57 words; Friendly Mentor at **+1.50** produced 55. A 2.70 alpha swing moved length by two words — under the noise floor, so length there is set by the model and the cap, not the dial.
- **Task-shaped prompt, same alpha, visible cost.** At **+1.50** on the code-review prompt the arm ran ~47 words of prose against the baseline's ~29, and the extra words pushed the deliverable out of the budget entirely.
- **Explainer prompt, same alpha, no difference at all.** At **+1.50** on the p-value prompt both arms ran to the identical 81 tokens with token-identical prose apart from two word choices. Length did not move.

So the verbosity vector is not inert, but it is not a length control either: it adds words only where the model already has room to elaborate, adds none where its continuation is committed, and never adds content. On capped budgets it converts wordiness into truncation. Report it as **unvalidated in the useful direction** — it should not be presented as working until a run shows added detail surviving to a complete answer.

### Known limitations

- **Six judged pairs: two Direct Architect, three Friendly Mentor, one Brutal Reviewer**, all `phi-3-mini-4k-instruct` on CPU. No cross-model runs.
- **One prompt per trait configuration.** Runs 3 and 5 share the same weights on different prompts and disagree completely (a small win versus a tie), so a single pair per configuration cannot establish an effect size — the trend across prompts is the signal, not any one run.
- **The sampling cap varied between runs** (120 tokens for Runs 1–3, 80 for Runs 4–6), so completeness scores are not comparable across runs — only within a pair.
- **Calibration is the ceiling.** The trait vectors were derived from contrastive sets of 5–6 pairs, which bounds how finely a slider can resolve. `empathy` and `initiative` have no vectors and are inert regardless of slider position.
- **Length confound.** The directness contrastive pairs differ in length, so part of that vector encodes brevity as well as bluntness. Verbosity was calibrated from a single contrastive set and is the least reliable of the three traits.
- **Every run truncated at `max_tokens`**, and the cut always landed on the payload rather than the setup. Completeness therefore cannot separate the arms, and the word-versus-detail judgement in the cross-run note is a hand reading of the prose, not a metric.
- **Memory tag leakage is alpha-dependent, not fixed.** Run 2 (+2.70 directness) echoed a single note as roughly the first 26 tokens of the answer. Run 3 (−1.20) and Run 4 (−1.20) paraphrased instead of quoting. Run 6 (+3.00) recites **both** notes in full, tags and bullets intact, across ~54% of the output budget. The same code path therefore shows three different behaviours, so no single statement about memory handling is true for all profiles: at negative alphas the notes are woven in, at +2.70 they are partially echoed, and at +3.00 they are reproduced outright.
- The judge is a single-pass reasoned evaluation, not a trained reward model — treat the scores as an ordering signal between arms, not an absolute quality metric.

---

## Held-Out Pair Evaluation — 300 pairs, every profile

Beyond the judged runs above, the presets were also measured quantitatively on **300 held-out contrastive pairs** (100 each for directness, formality and verbosity). For every pair the trait-positive response is scored under each profile and compared with the unsteered baseline, with a sign test on direction agreement.

| Profile | Mean Δ (nats/token) | Directness | Formality | Verbosity |
|---|---|---|---|---|
| Direct Architect | +0.048 | 68/100 ✔ | 78/100 ✔ | 30/100 ✘ |
| Friendly Mentor | −0.041 | 60/100 | 85/100 ✔ | 26/100 ✘ |
| Brutal Reviewer | +0.106 | 91/100 ✔ | spillover | 27/100 ✘ |

Directness and formality steer in the dialled direction; **verbosity steers against its label in all three profiles**. The protocol, per-trait tables, reproduction steps and limitations are in **[EVALUATION.md](EVALUATION.md)**, including the contradiction between the strongest directness effect here and the prose judges in Runs 1 and 6.

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

---

## Local API Integration (Universal Client Support)

MBCT provides a standard **OpenAI-compatible API layer** alongside a simplified direct REST API. Any third-party application, local script, or agent framework can connect directly to MBCT with zero code changes.

### Supported Integrations
* **OpenAI SDK** (`Python`, `Node.js`, `TypeScript`)
* **LangChain & LlamaIndex** (via `ChatOpenAI(base_url="http://127.0.0.1:8088/v1")`)
* **Developer Tools**: Continue.dev, Cursor, Open WebUI, LibreChat, LM Studio
* **Standard HTTP / cURL**: Any language supporting HTTP requests

### 1. OpenAI Python SDK Example
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8088/v1",
    api_key="mbct-local"  # Any dummy key is accepted
)

# Standard chat completion with optional trait steering
response = client.chat.completions.create(
    model="phi-3-mini",
    messages=[
        {"role": "system", "content": "You are a concise technical advisor."},
        {"role": "user", "content": "Should our startup adopt Kubernetes?"}
    ],
    temperature=0.3,
    max_tokens=100,
    extra_body={
        # Dial personality on the fly for this specific request!
        "traits": {"directness": 0.95, "formality": 0.80, "verbosity": 0.20},
        "inject_memory": True  # Uses external memory store
    }
)

print(response.choices[0].message.content)
```

### 2. cURL Example
```bash
curl -X POST http://127.0.0.1:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-3-mini",
    "messages": [{"role": "user", "content": "Explain mutexes in one sentence."}],
    "traits": {"directness": 0.9, "formality": 0.7}
  }'
```

### 3. Simple REST API (`/api/generate`)
```bash
curl -X POST http://127.0.0.1:8088/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What database architecture do we use?",
    "inject_memory": true
  }'
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | **OpenAI-compatible Chat Completions** (streaming + non-streaming, per-request trait overrides, memory injection) |
| `POST` | `/v1/completions` | **OpenAI-compatible Text Completions** |
| `GET` | `/v1/models` | **OpenAI-compatible Model Listing** |
| `POST` | `/api/generate` | **Simplified Local Generation API** (prompt, traits, memory) |
| `POST` | `/api/chat` | **Simplified Local Chat API** |
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
