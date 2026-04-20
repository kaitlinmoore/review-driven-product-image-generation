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
| `structured_extract` | `extract_structured_features.py` | `json` |
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
