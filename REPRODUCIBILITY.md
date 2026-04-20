# Reproducibility

How to reproduce the canonical pipeline run, what's deterministic, what
isn't, and the model versions pinned for comparability.

## Two Operating Modes

The project is being structured to support two modes:

- **Record mode (live API calls).** Runs the pipeline against live OpenAI,
  HuggingFace, and diffusion APIs. Produces and saves all intermediate
  artifacts. Default behavior.
- **Replay mode (offline, no API costs).** Reads cached call results from
  `replay_cache/` instead of hitting any external API. Lets a grader (or
  anyone on a different environment) reproduce the canonical results without
  incurring API costs or needing the gated Mistral weights.

The replay wrapper design lives in replay.py. It defines:

```python
REPLAY_DIR = Path('replay_cache')
REPLAY_MODE = os.environ.get("REPLAY_MODE", "record")  # 'record' or 'replay'

def cached_call(fn_name, inputs, live_fn, output_encoder, output_decoder):
    # record: call live_fn, save JSON to replay_cache/{fn_name}_{key}.json
    # replay: read cache, raise if missing
```

Keys are `SHA256(json(fn_name + inputs))[:16]`. **Extraction into `src/replay.py`
and wiring around the LLM / diffusion / evaluation calls is pending** — the
scaffolding in `replay_cache/` is in place so the wrapper can drop in without
breaking layout assumptions.

## Deterministic vs. Stochastic

| Stage | Behavior |
|---|---|
| Raw data pull (`pull_product_data.py`) | Deterministic given the UCSD Amazon-2023 snapshot URLs |
| Filter gate (`build_filter_cache.py`) | `gpt-4o-mini`, `temperature=0`, JSON mode — deterministic outputs per review |
| Preprocessing (`preprocess_reviews.py`) | Pure Python, deterministic |
| Prompt-context assembly (`build_prompt_context.py`) | Deterministic |
| Initial prompt (`generate_initial_prompt.py`) | `gpt-4o`, `temperature=0` — deterministic |
| Prompt refinement (`stepPrompt`) | Mistral-7B with `do_sample=True, temperature=0.7` — **stochastic** |
| Prompt rating (`ratePrompt`) | Mistral-7B with `do_sample=False` — deterministic given same input |
| Image generation — FLUX.1-schnell (`gen_image_flux.py`) | No fixed seed passed — **stochastic** |
| Image generation — gpt-image-1.5 (`gen_image_gpt.py`) | No seed param exposed by the API — **stochastic** |
| Image similarity (`compImage`) | CLIP / DINOv2 / SigLIP cosine — deterministic given fixed inputs |

Replay mode makes the stochastic stages deterministic by replaying the exact
cached result from the original record run.

## Pinned Model Versions (Canonical Run)
We explored multiple models during development, and these represent the final design inclusions.

| Role | Model ID |
|---|---|
| Filter gate | `gpt-4o-mini` (OpenAI API, `temperature=0`) |
| Initial prompt | `gpt-4o` (OpenAI API, `temperature=0`) |
| Prompt refinement + rating | `mistralai/Mistral-7B-Instruct-v0.3` (4-bit via `bitsandbytes`) |
| Image generation (model 1, open-weights) | `black-forest-labs/FLUX.1-schnell` via `huggingface_hub.InferenceClient(provider='fal-ai')`; 1024×1024, 4 inference steps |
| Image generation (model 2, proprietary) | `gpt-image-1.5` (OpenAI API); 1024×1024, `quality='medium'` |
| Evaluation — CLIP | `sentence-transformers/clip-ViT-B-32` |
| Evaluation — DINOv2 | `facebook/dinov2-base` |
| Evaluation — SigLIP | `google/siglip-base-patch16-224` |

## What a Re-Runner Needs

**For record mode:**
- OpenAI API key with GPT-4o + gpt-image-1.5 access (image-gen endpoints may require org verification).
- HuggingFace token. Serves two roles: (1) authenticates the gated
  Mistral-7B-Instruct-v0.3 download; (2) routes FLUX.1-schnell calls to
  fal-ai via `huggingface_hub.InferenceClient`.
- CUDA-capable GPU with ~6 GB VRAM for the 4-bit quantized Mistral. Image
  generation itself runs remotely (fal-ai for FLUX, OpenAI for gpt-image-1.5)
  so no local VRAM is needed for that stage.

**For replay mode:** Only the repository contents and Python 3.13+. No API
keys, no GPU, no gated-model access. *Pending wrapper extraction.*

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

---

*Draft — some sections (especially replay-mode instructions) will be filled
in when the wrapper is wired.*
