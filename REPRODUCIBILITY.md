# Reproducibility

How to reproduce the canonical pipeline run, what's deterministic, what
isn't, and the model versions pinned for comparability.

## Two Operating Modes

The pipeline supports two modes, selected by the `REPLAY_MODE` environment
variable:

- **Record mode (`REPLAY_MODE=record`, default).** Runs the pipeline against
  live OpenAI, HuggingFace, and fal-ai APIs, and against local Mistral /
  CLIP / DINOv2 / SigLIP. Every external call is also cached to
  `replay_cache/`. On subsequent runs with identical inputs, the cache is
  returned directly — no live call, no cost.
- **Replay mode (`REPLAY_MODE=replay`).** Reads cached call results from
  `replay_cache/` only. Raises `FileNotFoundError` if a cache entry is
  missing. Lets a grader (or anyone on a different environment) reproduce
  the canonical results with no API keys, no GPU, and no gated-model access.
- **Force-record mode (`REPLAY_MODE=force_record`).** Always calls live and
  overwrites the cache. Used to re-record specific calls after editing a
  prompt or upgrading a model.

The wrapper is implemented in `src/replay.py` as `cached_call`:

```python
from replay import cached_call, path_hash

def _live():
    # body that calls the live API / model
    return result

result = cached_call(
    fn_name='initial_prompt',      # appears in the cache filename
    inputs={                       # everything that affects the result
        'model': 'gpt-4o',
        'temperature': 0,
        'prompt_version': 'v1',
        'prompt_sha256': '...',
        'context_text': '...',
    },
    live_fn=_live,
    format='json',                 # 'json' | 'text' | 'png' | 'bytes'
)
```

Keys are `SHA256(json({'fn': fn_name, 'inputs': inputs}, sort_keys=True))[:16]`.
When an input is a local file path, wrap it with `path_hash()` so the key is
stable across machines with different absolute paths. System-prompt strings
are hashed into the inputs dict so prompt edits invalidate the cache
automatically.

All six production call sites are wrapped (see "What Gets Cached" below).

## Experimental Design — No Ground-Truth Leakage

The research question is: given only what customers *said* about a product,
how closely can a generative pipeline reconstruct what the product *looks*
like? This framing only holds if ground-truth product images (and any
features derived from them) are never used as a feedback signal inside the
generation loop. The current pipeline is structured to respect that
boundary.

**Ground-truth artifacts flow one direction only — into evaluation:**

| Script | Reads ground-truth? | Why it's safe |
|---|---|---|
| `generate_initial_prompt.py` | No | Reads `prompt_context.txt` (metadata + filtered reviews) |
| `PromptWriter.stepPrompt` | No | Reads review batches from the DataLoader |
| `PromptWriter.ratePrompt` | No | Mistral self-scores prompt descriptiveness |
| `agent_loop.agentLoop` | Title only | Uses `evalImage(img, ground_truth_text)` where `ground_truth_text` is the product **title string** from metadata — not image-derived |
| `gen_image_flux.py` / `gen_image_gpt.py` | No | Prompt-only inputs |
| `extract_structured_features.py --source ground_truth` | Yes | **Evaluation artifact.** Writes JSON to disk; no generation-path script reads its output |

The title-string signal inside `agentLoop` is considered fair: any real-world
caller of a text-to-image pipeline has *some* name for the product they want
to generate, and the title is the minimal identifying string. It is
metadata, not image-derived.

**Status.** As of 2026-04, we are awaiting course-staff guidance on whether
ground-truth-informed refinement would be considered fair game for this
assignment. The pipeline is structured so an opt-in variant (e.g. a
`stepPromptWithGT` method that takes a ground-truth feature record alongside
the review batch) could be added without rearchitecting. **If the design
changes, update this section, the `stepPrompt` row in the table below, and
any affected Q2/Q3 report framing.**

## Deterministic vs. Stochastic

| Stage | Behavior |
|---|---|
| Raw data pull (`pull_product_data.py`) | Deterministic given the UCSD Amazon-2023 snapshot URLs |
| Filter gate (`build_filter_cache.py`) | `gpt-4o-mini`, `temperature=0`, JSON mode — deterministic outputs per review |
| Preprocessing (`preprocess_reviews.py`) | Pure Python, deterministic |
| Prompt-context assembly (`build_prompt_context.py`) | Deterministic |
| Initial prompt (`generate_initial_prompt.py`) | `gpt-4o`, `temperature=0` — deterministic |
| Structured feature extraction (`extract_structured_features.py`) | `gpt-4o`, `temperature=0`, strict JSON schema — deterministic |
| Prompt refinement (`PromptWriter.stepPrompt`) | Mistral-7B with `do_sample=True, temperature=0.7` — **stochastic** |
| Prompt rating (`PromptWriter.ratePrompt`) | Mistral-7B with `do_sample=False` — deterministic given same input |
| Image generation — FLUX.1-schnell (`gen_image_flux.py`) | No fixed seed passed — **stochastic** |
| Image generation — gpt-image-1.5 (`gen_image_gpt.py`) | No seed param exposed by the API — **stochastic** |
| Image-vs-text similarity (`compImage`) | CLIP cosine — deterministic given fixed inputs |
| Image-vs-image similarity (`clip_image_similarity`, `dinov2_similarity`, `siglip_similarity`) | CLIP / DINOv2 / SigLIP cosine — deterministic given fixed inputs |

Replay mode makes the stochastic stages deterministic by returning the
exact cached result from the original record run.

## Pinned Model Versions (Canonical Run)

We explored multiple models during development; these are the final design
inclusions.

| Role | Model ID |
|---|---|
| Filter gate | `gpt-4o-mini` (OpenAI API, `temperature=0`) |
| Initial prompt + structured extraction | `gpt-4o` (OpenAI API, `temperature=0`) |
| Prompt refinement + rating | `mistralai/Mistral-7B-Instruct-v0.3` (4-bit via `bitsandbytes`) |
| Image generation (model 1, open-weights) | `black-forest-labs/FLUX.1-schnell` via `huggingface_hub.InferenceClient(provider='fal-ai')`; 1024×1024, 4 inference steps |
| Image generation (model 2, proprietary) | `gpt-image-1.5` (OpenAI API); 1024×1024, `quality='medium'` |
| Evaluation — CLIP | `sentence-transformers/clip-ViT-B-32` (image-vs-text) + `openai/clip-vit-base-patch32` (image-vs-image) |
| Evaluation — DINOv2 | `facebook/dinov2-base` |
| Evaluation — SigLIP | `google/siglip-base-patch16-224` |

## What Gets Cached

Every cache filename is `{fn_name}_{key}.{ext}`. The `fn_name` tags below
are the ones currently in use:

| `fn_name` | Source | Format |
|---|---|---|
| `filter_gate` | `build_filter_cache.py` | `json` |
| `initial_prompt` | `generate_initial_prompt.py` | `json` |
| `structured_features` | `extract_structured_features.py` | `json` |
| `step_prompt` | `PromptWriter.stepPrompt` | `text` |
| `rate_prompt` | `PromptWriter.ratePrompt` | `json` |
| `flux_gen` | `gen_image_flux.py` | `png` |
| `gpt_image_gen` | `gen_image_gpt.py` | `png` |
| `eval_clip_text` | `eval_image.compImage` | `json` |
| `eval_clip_image` | `eval_image.clip_image_similarity` | `json` |
| `eval_dinov2` | `eval_image.dinov2_similarity` | `json` |
| `eval_siglip` | `eval_image.siglip_similarity` | `json` |

Evaluation caches are particularly useful for replay mode: the cached float
is returned without loading the underlying CLIP / DINOv2 / SigLIP weights,
so a grader on a CPU-only laptop can reproduce all similarity numbers
without downloading ~2 GB of model checkpoints.

## What a Re-Runner Needs

**For record mode:**
- OpenAI API key with GPT-4o + gpt-image-1.5 access (image-gen endpoints may
  require org verification).
- HuggingFace token. Serves two roles: (1) authenticates the gated
  Mistral-7B-Instruct-v0.3 download; (2) routes FLUX.1-schnell calls to
  fal-ai via `huggingface_hub.InferenceClient`.
- CUDA-capable GPU with ~6 GB VRAM for the 4-bit quantized Mistral. Image
  generation itself runs remotely (fal-ai for FLUX, OpenAI for gpt-image-1.5)
  so no local VRAM is needed for that stage.

**For replay mode:** Only the repository contents and Python 3.13+. Set
`REPLAY_MODE=replay` in the environment and run the same script sequence
as record mode. No API keys, no GPU, no gated-model access.

## Environment Setup — GPU Wheel Selection

`pip install -r requirements.txt` does NOT automatically give you a
CUDA-enabled PyTorch build. The `torch>=2.1` pin is loose, and PyPI's
default torch wheel is CPU-only. Running the agent loop with a CPU-only
wheel raises `AssertionError: Torch not compiled with CUDA enabled` the
moment `bitsandbytes` tries to place the quantized Mistral model on the
GPU.

The fix is to install torch (and torchvision, which `transformers` pulls
in transitively) from PyTorch's platform-specific wheel index. For modern
NVIDIA GPUs:

| GPU generation | Compute capability | Recommended CUDA index |
|---|---|---|
| Ada Lovelace (RTX 40-series) | sm_89 | `cu124` or `cu126` |
| Hopper (H100/H200) | sm_90 | `cu124` or `cu126` |
| Blackwell (RTX 50-series, B100) | sm_120 | `cu128` |

For the RTX 5080 / 5090 family specifically, `cu128` is required — earlier
CUDA builds predate sm_120 support. The canonical install recipe:

```powershell
# PowerShell (Windows)
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

```bash
# bash (Linux / macOS-with-eGPU)
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

Verify the install picked up CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# expected: 2.11.0+cu128 True NVIDIA GeForce RTX 5080 ...
```

**If `torch.cuda.is_available()` returns False** after this install, check
for a shadowing install in `C:\Program Files\Python313\Lib\site-packages\`
(Windows system-site) or `/usr/lib/python3/dist-packages/` (Linux). User-site
installs take precedence only if user-site appears first in `sys.path`,
which is the default but occasionally gets overridden. Run
`python -c "import torch; print(torch.__file__)"` to confirm which location
is actually loaded.

Other packages that commonly fail silently on a fresh install because they
are transitive dependencies not all pins catch:

- `sentencepiece` — tokenizer backend for Mistral (Llama family). Listed in
  `requirements.txt`.
- `accelerate` — needed by `transformers` for any device-mapped load.
  Listed in `requirements.txt`.
- `bitsandbytes` — 4-bit quantization. Listed in `requirements.txt`.

Run this check after `pip install -r requirements.txt` to confirm nothing
is missing:

```bash
python -c "
import importlib
for m in ['torch','transformers','accelerate','bitsandbytes','sentencepiece',
         'huggingface_hub','PIL','sentence_transformers','openai','dotenv']:
    try: importlib.import_module(m); print('OK  ', m)
    except Exception as e: print('FAIL', m, e)
"
```

## Known Non-Determinism in the Canonical Run

- `FLUX.1-schnell` supports a `seed` parameter but we don't currently pin it;
  subsequent record runs at the same prompt will produce different images.
  Showing the variance helps characterize the generation stage; pinning a
  seed is a candidate `--seed` flag in future work.
- `gpt-image-1.5` does not expose a seed parameter at all, so its output is
  API-level stochastic. No way to make it deterministic at the record stage;
  replay mode is the only path to reproducibility for this model.
- `stepPrompt` uses `do_sample=True, temperature=0.7` — the prompt trajectory
  will differ between record runs.

In all three cases, replay mode pins the output to whatever was observed on
the canonical record run.

## Ground-Truth Image Selection

`extract_structured_features.py --source ground_truth` aggregates multiple
product photos into a single feature record. Not all images in
`data/{product}/images/` are appropriate for this (some are infographics,
packaging shots, or lifestyle scenes that would contaminate the feature
extraction). The exclusion list lives in `GROUND_TRUTH_IMAGE_EXCLUDES` at
the top of `extract_structured_features.py` and is per-product. Graders
reproducing via replay mode do not need to touch this — the cached
`structured_extract` entries already reflect the canonical exclusion list.
