# Review-Driven Product Image Generation

This pipeline reconstructs product images from the product metadata, description,
and customer reviews, with no direct image input. Amazon product reviews pass through an LLM
visual-content filter, feed an LLM that writes a free-form visual description,
and that description drives a diffusion model. Iterative refinement is used for both prompt tuning and image tuning. Generated images are then compared to the real product images via learned-embedding similarity (CLIP + DINOv2 + SigLIP) to quantify how close the pipeline got.

> **Research Question:** given only what customers *said* about a
> product, how closely can a generative pipeline reconstruct what the product
> actually *looks* like?

## Attribution

CMU Heinz 94-844 Generative AI Lab, Spring 2026. Team of 5.

An AI-use disclosure documenting how AI tools contributed to the work
is in [`docs/AI_USE.md`](docs/AI_USE.md).

## Repository Layout

```
.
├── src/                      # production scripts (run the pipeline end-to-end)
├── exploration/              # methodology artifacts (candidate scoring, gate experiment)
├── data/
│   ├── filter_caches/        # cached LLM gate decisions — committed, scientific
│   └── {product}/            # per-product metadata, images, intermediate outputs
├── replay_cache/             # cached API / model outputs for offline re-runs (see REPRODUCIBILITY.md)
├── docs/
│   ├── PROMPTS.md            # all five LLM prompts with SHA-256 hashes
│   ├── decisions_log.md      # methodology decisions and rationale
│   ├── artifact_map.md       # per-question evidence map
│   ├── pipeline_diagram.png  # agent architecture diagram
│   └── AI_USE.md             # AI-use disclosure
├── requirements.txt
├── REPRODUCIBILITY.md
└── README.md
```

## Quick Start

```bash
# 1. Clone.
git clone <repo-url>
cd review-driven-product-image-generation

# 2. Create and activate a venv.
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# bash:       source .venv/bin/activate

# 3. Install dependencies.
pip install -r requirements.txt

# 4. Add API keys to a .env file in the repo root:
#    OPENAI_API_KEY=...        # filter-gate, initial prompt, structured extraction, gpt-image-1.5 generation
#    HUGGINGFACE_TOKEN=...     # gated Mistral-7B-Instruct-v0.3 download + FLUX.1-schnell (routed via fal-ai)

# 5. Pull raw review data (reproducible from the UCSD Amazon-2023 dump).
python src/pull_product_data.py

# 6. Run the preprocessing and prompt-assembly pipeline.
python src/build_filter_cache.py --workers 8
python src/preprocess_reviews.py
python src/build_prompt_context.py
python src/generate_initial_prompt.py

# 7. Extract structured visual features for the five reference comparison
#    sources. Feeds Q2/Q3 evaluation.
python src/extract_structured_features.py --source initial
python src/extract_structured_features.py --source prompt_context
python src/extract_structured_features.py --source metadata_only
python src/extract_structured_features.py --source reviews_only
python src/extract_structured_features.py --source ground_truth

# 8. Run the agent loop (iterative refinement + image generation) for the
#    three ablation configurations. Each writes per-config artifacts plus
#    canonical pointers. See REPRODUCIBILITY.md "Agent-Loop Run Configs".
python src/run_agent_pipeline.py --config-name v1_title_clip
python src/run_agent_pipeline.py --config-name v2_initial_prompt_clip \
    --quality-signal clip_text --reference initial_prompt
python src/run_agent_pipeline.py --config-name v3_initial_prompt_features \
    --quality-signal structured_features --reference initial_prompt \
    --quality-target 0.7

# 9. Extract structured features from each agent-loop output (per-config
#    converged prompts and generated images), for ablation comparison.
for product in backpack chess_set espresso_machine headphones jeans water_bottle; do
  for model in flux gpt; do
    for config in v1_title_clip v2_initial_prompt_clip v3_initial_prompt_features; do
      python src/extract_structured_features.py --source converged \
          --only $product --config-name $config --model $model
      python src/extract_structured_features.py --source generated \
          --only $product --config-name $config --model $model
    done
  done
done
```

After these steps, each product directory contains the full set of
artifacts: the initial prompt, the per-config refined prompts and
generated images from each agent-loop run, and structured-feature
extractions from each comparison source (initial, ground_truth,
metadata-only, reviews-only, prompt_context, plus per-config converged
and generated). The `data/{product}/` files are flat (not in a
subdirectory) and named by source and config tag.

### GPU users: install CUDA-enabled PyTorch

`pip install -r requirements.txt` pulls the CPU-only PyTorch wheel from
PyPI. The image-generation + refinement loop (Phase 3) needs CUDA-enabled
torch to drive the local 4-bit Mistral at speed. Install the matching
CUDA wheel *after* the requirements step:

```powershell
# PowerShell (Windows); replace cu128 with cu126/cu124 per your GPU
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

```bash
# bash (Linux)
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) "Environment Setup — GPU Wheel
Selection" for the GPU-generation → CUDA-version mapping and a verification
one-liner.

### Replay mode (no API costs, no GPU)

Every external call (OpenAI, HuggingFace, FLUX via fal-ai, local Mistral,
CLIP / DINOv2 / SigLIP) is wrapped in a replay cache. To rerun the canonical
pipeline offline from the committed cache, set `REPLAY_MODE=replay` before
running any of the scripts above:

```bash
# PowerShell: $env:REPLAY_MODE = 'replay'
# bash:       export REPLAY_MODE=replay
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for details on record vs. replay
semantics and the `force_record` mode used when re-recording specific calls.

## Pipeline Overview

```
raw reviews       ──►  LLM gate (gpt-4o-mini)     ──►  filter_decisions_v1.json
                                                        │
reviews_ranked    ◄─── dedupe + hygiene + gate ◄────────┘
  + metadata      ──►  prompt_context.txt  ──►  GPT-4o  ──►  initial_prompt.txt
                                                                │
                                                                ▼
                                              PromptWriter iteration loop
                                              (Mistral-7B stepPrompt/ratePrompt)
                                                                │
                                                                ▼
                                    genImage (FLUX.1-schnell | gpt-image-1.5) → generated image
                                                                │
                                                                ▼
                                    evaluation: CLIP + DINOv2 + SigLIP cosine
                                    similarity vs. ground-truth product photos
```

**Key Models:**

- Filter gate: `gpt-4o-mini` (temperature 0, JSON mode)
- Initial prompt + structured extraction: `gpt-4o` (temperature 0)
- Prompt refinement + rating: `mistralai/Mistral-7B-Instruct-v0.3` (4-bit quantized)
- Image generation (open-weights): `black-forest-labs/FLUX.1-schnell` via HuggingFace `InferenceClient(provider='fal-ai')`
- Image generation (proprietary): `gpt-image-1.5` via OpenAI API
- Evaluation: `sentence-transformers/clip-ViT-B-32`, `facebook/dinov2-base`, `google/siglip-base-patch16-224`

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Record vs. replay modes, what's
deterministic vs. stochastic, and exact model versions used for the canonical
run.

