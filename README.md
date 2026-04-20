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

An AI-use disclosure at the end of the final report details how AI tools
contributed to the work.

## Repository Layout

```
.
├── src/                      # production scripts (run the pipeline end-to-end)
├── notebooks/                # development artifacts; not the runnable path
├── exploration/              # methodology artifacts (candidate scoring, gate experiment)
├── data/
│   ├── filter_caches/        # cached LLM gate decisions — committed, scientific
│   └── {product}/            # per-product metadata, images, intermediate outputs
├── replay_cache/             # scaffolding for offline re-runs (design landed, wrapper pending)
├── docs/                     # candidate-product documentation
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
```

After these steps, each product directory contains an `initial_prompt.txt`
ready for the image-generation + refinement loop.

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

---

*DRAFT*
