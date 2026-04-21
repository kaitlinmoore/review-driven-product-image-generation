# Pipeline Overview

## What the Pipeline Does

The pipeline takes Amazon customer reviews of a product and tries to
reconstruct what the product looks like, **without ever showing it the
product's actual photos during generation**. The reviews and metadata
go through several stages (filtering, prompt construction, iterative
refinement, image generation), and the final generated images are then
compared against the real product photos to measure how close the
reconstruction got.

The general research question: **Given only what customers SAID about a
product, how closely can a generative pipeline reconstruct what the
product LOOKS like?**

## End-to-End Flow

A static rendering is available at
[`pipeline_diagram.png`](pipeline_diagram.png) for sharing in slides
or other contexts where Mermaid won't render. The Mermaid source below
is the canonical version. When viewing this file on GitHub, it
renders inline and stays in sync with future edits.

```mermaid
flowchart TD
    A[UCSD Amazon Reviews 2023<br/>raw category dumps] -->|pull_product_data.py| B[reviews.jsonl<br/>+ metadata.json<br/>+ product images]

    B -->|build_filter_cache.py<br/>gpt-4o-mini per-review classification| C[filter_decisions_v1.json<br/>pass/fail per review]

    C -->|preprocess_reviews.py<br/>dedupe + rank by helpfulness| D[reviews_ranked.jsonl<br/>top visually-rich reviews]

    D -->|build_prompt_context.py<br/>combine metadata + top 30 reviews| E[prompt_context.txt]

    E -->|generate_initial_prompt.py<br/>gpt-4o single call| F[initial_prompt.txt<br/>~250-word visual description]

    F -->|extract_structured_features.py<br/>--source initial / metadata_only / etc.| G[structured_features_*_v1.json<br/>13-field reference dicts]

    F -->|run_agent_pipeline.py<br/>iterative refinement + image gen| H[converged_prompt + generated_image<br/>per product, per image model]

    H -->|extract_structured_features.py<br/>--source converged / generated| I[structured_features_<br/>converged/generated_v1.json]

    J[product images<br/>data/{product}/images/] -->|extract_structured_features.py<br/>--source ground_truth| K[structured_features_<br/>ground_truth_v1.json]

    G -.->|compared against| K
    I -.->|compared against| K
    K -.->|reference for evaluation| L((Q1/Q2/Q3/Q4<br/>findings))
    G -.-> L
    I -.-> L

    style J fill:#fde7e7
    style K fill:#fde7e7
    style L fill:#e7f3fd
```

The dotted arrows in the bottom block represent **evaluation comparisons**,
not data flow. Ground-truth artifacts (red) only feed evaluation, never
the generation loop.

## Stage by Stage

The pipeline has six logical stages. Each runs as its own script in
`src/` and produces specific artifacts that the next stage consumes.

### Stage 0 | Data Acquisition

**Script**: `src/pull_product_data.py`
**Inputs**: Six product `parent_asin` codes (Amazon product IDs)
**Outputs per product**:
- `data/{product}/reviews.jsonl` | raw customer reviews
- `data/{product}/metadata.json` | product title, description, features
- `data/{product}/images/` | reference product photos
- `data/{product}/summary.json` | counts and a quick-look reference

Streams directly from the UCSD McAuley Lab Amazon Reviews 2023 dataset
without downloading the full 50+ GB category files. Each product is
isolated by parent_asin during streaming.

### Stage 1 | Visual-Content Filtering

**Script**: `src/build_filter_cache.py`
**Input**: `reviews.jsonl` per product
**Output**: `data/filter_caches/{product}_filter_decisions_v1.json`

Each review is classified by gpt-4o-mini as either *visually
descriptive* (talks about colors, materials, shapes, finishes,
appearance) or *not visually descriptive* (only covers shipping,
function, price, recipient reactions, etc.). Reviews are processed in
parallel with strict JSON-mode output. Includes a brief reason string
per decision.

The filter decisions are cached by review-content hash so re-runs are
free. Filter cache is committed to git as methodological artifact.

### Stage 2 | Preprocessing and Prompt-Context Assembly

**Scripts**: `src/preprocess_reviews.py` → `src/build_prompt_context.py`

Preprocessing: deduplicates reviews by `(user_id, timestamp)`, joins
each review with its filter decision, ranks by `helpful_vote` (with
ties broken by text length), produces `reviews_ranked.jsonl`.

Prompt-context assembly: takes the top 30 visually-descriptive reviews
plus the metadata block (title, description, features) and formats
them into a single text block (`prompt_context.txt`) that gpt-4o will
read in the next stage.

### Stage 3 | Initial Prompt Generation

**Script**: `src/generate_initial_prompt.py`
**Input**: `prompt_context.txt`
**Output**: `data/{product}/initial_prompt.txt`

A single gpt-4o call per product. The system prompt instructs the
model to write a roughly 150-300 word visual description suitable for
text-to-image diffusion. The output is the foundation that all
subsequent refinement builds on.

This is a one-shot stage with no iteration, no comparison. Just
"distill the metadata and filtered reviews into a visual description."

### Stage 4 | Iterative Refinement and Image Generation (the Agent Loop)

**Script**: `src/run_agent_pipeline.py`
**Input**: `initial_prompt.txt`, `reviews_ranked.jsonl` (for refinement
batches), `metadata.json` (for product title)
**Outputs per (product, image-model, config)**:
- `converged_prompt_{model}_{config}.txt` | final refined prompt
- `generated_image_{model}_{config}.png` | best-iteration generated image
- `agent_run_{model}_{config}_meta.json` | full provenance trace

Plus canonical pointer files (written by default; suppressed with
`--no-promote`) that downstream stages can read without knowing which
config is currently blessed:
- `converged_prompt_{model}.txt`
- `generated_image_{model}.png`
- `converged_prompt.txt` (a copy of the FLUX-config converged prompt)

This is the most complex stage. For each (product, image-model,
config) combination:

1. **Outer loop** runs `image_count` image attempts (default: 3).
2. For each image attempt:
   - **Inner loop**: Mistral-7B refines the prompt across multiple
     iterations, each time stepping through a fresh batch of reviews
     and self-rating the result on a 0–100 descriptiveness scale.
   - **Image generation**: The refined prompt is sent to
     FLUX.1-schnell (model 1) or gpt-image-1.5 (model 2).
   - **Quality scoring**: The generated image is scored against a
     reference using the configured quality signal (CLIP cosine for
     v1/v2, structured-feature agreement for v3).
3. **Surrogate-projected adaptive iteration control**: Between image
   attempts, a quadratic surrogate is fit to observed
   (descriptiveness, quality) and (cumulative iterations, quality)
   data. The fit is used to project what threshold and iteration
   budget would be needed to reach the quality target, and the next
   image attempt's parameters are adjusted accordingly.
4. **Best image selection**: Across the image attempts, the one
   with the highest quality score is saved as the canonical artifact
   for that (product, model, config).

Two image models × six products × three configurations = 36 distinct
(product, model, config) runs.

### Stage 5 | Structured-Features Extraction (Multi-Source)

**Script**: `src/extract_structured_features.py`
**Inputs**: Various, depending on `--source` flag:
- `--source initial` → reads `initial_prompt.txt`
- `--source converged` → reads `converged_prompt.txt` (canonical pointer)
  or per-config `converged_prompt_{model}_{config}.txt` when
  `--config-name` and `--model` are also provided
- `--source prompt_context` → reads `prompt_context.txt`
- `--source metadata_only` → reads `metadata.json` (excludes reviews)
- `--source reviews_only` → reads top reviews (excludes metadata)
- `--source ground_truth` → reads multiple images from `images/`
- `--source generated` → reads a specified image (via `--image-path`)
  or per-config `generated_image_{model}_{config}.png` when
  `--config-name` and `--model` are provided

**Output**: `data/{product}/structured_features_{source}_v1.json`,
or `structured_features_{source}_{model}_{config}_v1.json` for
per-config extractions.

Each source is processed by gpt-4o (or gpt-4o vision, for image sources)
into the same 13-field structured-features schema. This produces a
common comparison structure across very different input types. It acts as an anchor and
bridge that lets evaluation say "here's what the initial prompt
described, here's what the ground truth actually shows, here's what
the generated image rendered, scored on the same fields."

### Evaluation | Comparison and Metric Computation

**Library**: `src/eval_image.py`

Provides three families of comparison metrics:
1. **Image-vs-text CLIP cosine** (`compImage`) | used inside the agent
   loop's quality signal for v1 and v2 configurations.
2. **Image-vs-image cosine** (`clip_image_similarity`,
   `dinov2_similarity`, `siglip_similarity`) | used post-hoc to
   compare each generated image against the ground-truth reference
   photos.
3. **Structured-feature agreement** (`feature_agreement`,
   `eval_structured_features`) | used inside the agent loop's quality
   signal for v3, and post-hoc to compare structured features across
   different sources.

## Code Organization

```
src/
├── pull_product_data.py           # Stage 0
├── build_filter_cache.py          # Stage 1
├── preprocess_reviews.py          # Stage 2 (part 1)
├── build_prompt_context.py        # Stage 2 (part 2)
├── generate_initial_prompt.py     # Stage 3
├── run_agent_pipeline.py          # Stage 4 (driver)
├── agent_loop.py                  # Stage 4 (orchestration logic)
├── prompt_writer.py               # Stage 4 (Mistral wrapper)
├── reviews_dataloader.py          # Stage 4 (DataLoader for refinement batches)
├── gen_image_flux.py              # Stage 4 (FLUX wrapper)
├── gen_image_gpt.py               # Stage 4 (gpt-image wrapper)
├── extract_structured_features.py # Stage 5
├── eval_image.py                  # Evaluation library
├── replay.py                      # Replay-cache wrapper used everywhere
└── __init__.py
```

Every external API call (OpenAI, fal-ai, HuggingFace gated downloads)
and every heavy local model inference (Mistral, CLIP, DINOv2, SigLIP)
is wrapped through `replay.py`'s `cached_call`. This makes the entire
pipeline reproducible offline once the cache is committed.

## Key Models

| Stage | Model | Used for |
|---|---|---|
| 1 | gpt-4o-mini | Per-review visual-content classification |
| 3 | gpt-4o | Initial prompt generation from metadata + reviews |
| 4 (inner loop) | Mistral-7B-Instruct-v0.3 (4-bit) | Prompt refinement and self-rating |
| 4 (image gen) | FLUX.1-schnell (open-weights, via fal-ai) | Image generation, model 1 |
| 4 (image gen) | gpt-image-1.5 (OpenAI) | Image generation, model 2 |
| 4 (eval, v1/v2) | CLIP (`clip-ViT-B-32`) | Image-vs-text quality signal |
| 4 (eval, v3) | gpt-4o vision | Per-image structured-feature extraction |
| 5 | gpt-4o vision | Multi-source structured-feature extraction |
| Post-hoc eval | CLIP (`openai/clip-vit-base-patch32`), DINOv2, SigLIP | Image-vs-image comparison vs. ground truth |

## Reproducibility Model

The pipeline supports two operating modes (set via the `REPLAY_MODE`
environment variable):

- **Record mode (default)** | runs against live APIs and local models;
  caches every external call. Re-runs with identical inputs read from
  the cache and skip the live call.
- **Replay mode** | reads from the cache only; raises an error on cache
  miss. A grader can clone the repository, set `REPLAY_MODE=replay`,
  and reproduce the canonical results without any API keys, GPU, or
  gated-model access.

The replay cache is committed to git so it travels with the repo.

See `REPRODUCIBILITY.md` at the repo root for full details on the
reproducibility model, the environment-setup gotcha for CUDA wheel
selection, and the artifact-versioning scheme for ablation runs.
