# Data Processing

How the project went from "the McAuley Lab Amazon Reviews 2023 dataset
exists" to "every product has the artifacts the agent loop needs."
Covers product selection through structured-features extraction.

## Product Selection

### Source Dataset

McAuley Lab Amazon Reviews 2023: a large public dataset of Amazon
product reviews and metadata, organized by product category. Hosted at
`mcauleylab.ucsd.edu`. Each category file is a gzipped JSONL (one JSON
record per line) with millions of review records.

### Selection Methodology

The candidate-search code is in `exploration/explore_candidates.py`.
It streams the gzipped category file directly over HTTP, decompresses
on the fly, and filters to records matching:
- A target popularity band (number of ratings)
- An optional keyword in the product title
- Has at least one usable image URL
- Has non-trivial description and features text

Output is a CSV per (category, popularity_band, keyword) combination,
listing candidate products with their `parent_asin` and image URLs.

The CSVs are committed under `exploration/candidates/`. They document
which products were considered for the project. Roughly 40 candidate
CSVs spanning Home_and_Kitchen, Toys_and_Games, Sports_and_Outdoors,
Clothing_Shoes_and_Jewelry, and Electronics.

### Selection Criteria for the Canonical Six

The six products selected for the canonical run were chosen to span:
- **Multiple categories** (per assignment requirement)
- **Multiple visual challenges**:
  - Simple geometry (water bottle, chess set)
  - Complex composition with text/logos (espresso machine, headphones)
  - Composed scenes with human context (jeans, hiking backpack)

The six final products:

| Slug | Brand and product | Category | Visual challenge |
|---|---|---|---|
| `water_bottle` | HYDRO CELL Stainless Steel Insulated | Home_and_Kitchen | Simple cylindrical form with branded text |
| `chess_set` | Chess Armory 15 Inch Wooden | Toys_and_Games | Intricate piece geometry, wood grain |
| `espresso_machine` | Breville Barista Express BES870XL | Home_and_Kitchen | Complex multi-component machine, dials, text labels |
| `headphones` | Beats Studio3 over-ear | Electronics | Iconic silhouette, prominent brand mark |
| `jeans` | Levi's Women's Mile High Super Skinny | Clothing_Shoes_and_Jewelry | Garment fit and texture; on-model photos |
| `backpack` | Venture Pal 40L Hiking Backpack | Sports_and_Outdoors | Functional details (straps, compartments, rainfly) |

### One Product Is a Special Case

The **backpack** has only 30 visually-descriptive reviews after filtering
(out of 813 raw reviews. Most backpack reviewers focus on durability
and function rather than appearance). The DataLoader for the agent loop
skips the top 30 reviews because they're already incorporated into the
initial prompt. After that skip, **zero reviews remain for refinement**.

This means the backpack's agent loop generates 3 copies of the
initial-prompt-generated image without any review-driven refinement.
Backpack therefore appears in the v1/v2/v3 results as a baseline case
(initial-prompt-only generation), not as an ablation contributor.

This is treated as a **finding, not a bug**. Not all products in the
canonical six have enough visually-rich reviews to refine on. Backpack
is the limit case demonstrating this.

## Raw Data Acquisition

`src/pull_product_data.py` produces, per product:

| File | Content | Tracked in git? |
|---|---|---|
| `data/{product}/reviews.jsonl` | Raw review records | No (gitignored b/c large, reproducible) |
| `data/{product}/metadata.json` | Product title, description, features, images URLs | Yes |
| `data/{product}/images/` | Reference product photos pulled from Amazon | Yes |
| `data/{product}/summary.json` | Counts and sanity-check info | Yes |

Image counts vary by product (1–9 images each).

## Visual-Content Filtering

`src/build_filter_cache.py` runs every review through gpt-4o-mini with
a strict-JSON system prompt asking "does this review describe how the
product looks?" Output is a binary classification plus a brief reason.

### Filter Pass Rates by Product

Of the products' raw reviews, what fraction passed the visual-content
gate:

| Product | Raw reviews | Pass rate | Available after dedup + skip-top-30 |
|---|---|---|---|
| chess_set | 1535 | ~33% | 477 reviews available for refinement |
| water_bottle | very large | high | 1529 available |
| jeans | ~1100 | ~22% | 244 available |
| espresso_machine | very large | mid | 278 available |
| headphones | 1628 | 6% | 65 available |
| backpack | 813 | 4% | **0 available** (the limit case) |

The pass rates reflect **how visually-focused each product's reviewer
community is**. Chess and water bottle reviewers describe appearance;
backpack reviewers focus on function.

### Cached and Committed

The filter decisions for each product are cached at
`data/filter_caches/{product}_filter_decisions_v1.json`. The cache file
includes a SHA256 of the system prompt baked into the header, so any
edit to the prompt automatically invalidates affected entries. These
cache files ARE committed to git. They are the canonical record of
which reviews were judged useful.

## Preprocessing and Ranking

`src/preprocess_reviews.py` produces `data/{product}/reviews_ranked.jsonl`.
Steps:

1. Load raw reviews; merge with filter decisions.
2. Deduplicate by synthesized id `(user_id, timestamp)`. (Amazon's
   dataset has 199 confirmed exact-text duplicates across the six
   products; deduplication is correct behavior.)
3. Drop reviews that failed the filter or have empty text.
4. Sort remaining reviews by `helpful_vote` descending (with text length
   as the tiebreaker for the long tail).
5. Write to `reviews_ranked.jsonl` in ranked order.

The first 30 reviews from this file feed the initial-prompt generation;
reviews 31+ feed the agent loop's refinement DataLoader (so the two
stages don't see overlapping content).

## Prompt Context Assembly

`src/build_prompt_context.py` reads the metadata and the top 30 ranked
reviews and produces `data/{product}/prompt_context.txt`. Format is a
plain text block with a metadata section followed by labeled review
excerpts. This file is the input to the next stage's gpt-4o call.

## Initial Prompt Generation

`src/generate_initial_prompt.py` makes a single gpt-4o call per
product. The system prompt (committed in the script as
`INITIAL_PROMPT_SYSTEM_V1`) instructs the model to write a 150-300
word visual description distilled from the metadata and reviews,
focused on what the product LOOKS like (not on how it functions or
ships).

Output: `data/{product}/initial_prompt.txt` (~250 words on average).

A provenance sidecar is written alongside:
`initial_prompt_meta.json` records the model, temperature, prompt
version SHA, token counts, cost, and timestamp. The same provenance
pattern is used throughout the pipeline.

## Structured-Features Extraction

`src/extract_structured_features.py` is a multi-source script that
runs gpt-4o (or gpt-4o vision) over different inputs and produces a
common 13-field JSON output:

| Field | Type | Example value (water_bottle ground_truth) |
|---|---|---|
| `product_type` | string | "water bottle" |
| `primary_colors` | list[string] | ["pink"] |
| `materials` | list[string] | ["stainless steel"] |
| `finish` | enum | "matte" |
| `texture` | enum | "smooth" |
| `size_descriptor` | enum | "medium" |
| `measurements` | string \| null | null |
| `shape_and_form` | string | "Cylindrical with a wide mouth and a loop handle on the lid." |
| `decorative_elements` | list[string] | ["logo"] |
| `visible_parts` | list[string] | ["lid", "handle"] |
| `brand_visibility` | enum | "prominent" |
| `brand_description` | string \| null | "HYDRO CELL text logo on the front in white." |
| `overall_aesthetic` | string | "A vibrant and sleek water bottle with a modern design." |

The extraction is run with `--source` flag pointing at one of seven
input types. Reference sources are extracted once per product. Per-config
sources (`converged`, `generated`) are extracted once per (product,
image-model, config) combination.

### Source Types

The structured-feature schema is extracted from seven different source
types per product, giving comparable outputs across very different
inputs. The sources fall into two categories:

**Reference sources** | extracted once per product, fixed across the
agent-loop ablation configs:

| Source | Input | What it represents |
|---|---|---|
| `initial` | `initial_prompt.txt` | The initial gpt-4o-generated visual description |
| `prompt_context` | `prompt_context.txt` | Metadata + top reviews combined (no LLM rewrite) |
| `metadata_only` | `metadata.json` | Product metadata alone (no reviews) |
| `reviews_only` | top reviews from `reviews_ranked.jsonl` | Reviews alone (no metadata) |
| `ground_truth` | All non-excluded images in `data/{product}/images/` | What the product actually looks like, per the reference photos |

**Per-config sources** | extracted per (image-model, config) for each
ablation run:

| Source | Input | What it represents |
|---|---|---|
| `converged` | `converged_prompt_{model}_{config}.txt` | The final refined prompt from one ablation run |
| `generated` | `generated_image_{model}_{config}.png` | The best-iteration image from one ablation run |

Reference sources establish the input-side comparison space ("what did
each input source describe the product as having?"). Per-config
sources establish the output-side ("what did each agent-loop run
produce?"). Direct comparison is straightforward because the schema is
the same across all sources.

### Ground-Truth Image Curation

Not all images in `data/{product}/images/` are appropriate for
ground-truth feature extraction. Some are infographics, packaging
shots, lifestyle scenes, or accessory-only photos that would
contaminate the feature extraction.

A per-product exclusion list lives in `GROUND_TRUTH_IMAGE_EXCLUDES`
at the top of `extract_structured_features.py`. For example:

- **water_bottle**: excludes 5 of 9 images (alt_02 through alt_06 are
  off-model variant displays)
- **headphones**: excludes 1 image (alt_05 is a case-only shot)
- **backpack, chess_set, espresso_machine, jeans**: all images
  included (no exclusions necessary)

The exclusion list is documented in REPRODUCIBILITY.md and is part of
the canonical record. Graders reproducing in replay mode get the
already-curated structured-features outputs without needing to re-do
the curation.

## Final Artifact State Per Product

After the full pipeline runs end-to-end, each product directory
contains artifacts from each stage:

**Source data (Phase 0):**
- `reviews.jsonl` (gitignored — large, regenerable)
- `metadata.json`, `images/`, `summary.json`

**Filter and prompt assembly (Phases 1–2):**
- `reviews_ranked.jsonl`, `prompt_context.txt`
- `initial_prompt.txt` and provenance sidecar

**Agent-loop output (Phase 3):**
For each (image-model, config) combination | 6 combinations per
product (2 models × 3 configs):
- `converged_prompt_{model}_{config}.txt` | final refined prompt (the
  best iteration)
- `generated_image_{model}_{config}.png` | best-iteration image
- `agent_run_{model}_{config}_meta.json` | provenance with hyperparams,
  iteration counts, descriptiveness and quality trajectories

Plus canonical pointers for downstream tooling that doesn't need to
know which config is "blessed":
- `converged_prompt_{model}.txt`, `generated_image_{model}.png` per model
- `converged_prompt.txt` (a copy of the FLUX-promoted converged prompt)

**Per-iteration trajectories (Phase 3 detail):**
For each (model, config) combination, the 3 per-image-attempt artifacts
(not just the best-selected one) are organized under
`data/{product}/trajectories/{model}_{config}/`:

    data/{product}/trajectories/{model}_{config}/
        iteration_0/
            prompt.txt           | the refined prompt used for this attempt
            generated_image.png  | the image produced
            metadata.json        | per-iteration score, descriptiveness, iters_taken
        iteration_1/ ...
        iteration_2/ ...

18 iteration folders per product (6 combinations × 3 iterations). Useful
for slide-deck trajectories and failure-mode spot-checks where the
numeric summary isn't enough and the writer wants to see the actual
prompt and image at each step. The script `src/extract_iteration_trajectories.py`
regenerates this layout from the timestamped `runs/` subdirectories
after a fresh agent-loop run.

**Structured-feature extractions (Stage 5):**
The 13-field schema is extracted from each of the seven source types
described above. Each extraction produces a paired feature file and
provenance sidecar:

- 5 reference sources × 1 extraction per product = 10 files per product
- 2 per-config sources × 6 (model, config) combinations × 2 files = 24
  files per product

Per-product file count after the full pipeline runs to completion is
roughly 60–70 files across all categories. Exact counts vary by
product. For example, the backpack runs every (model, config)
combination but each one generates the initial-prompt-only image three
times because no review pool is available for refinement (see the
"One Product Is a Special Case" note above), so backpack's generated
artifacts are valid but represent the same image across attempts.
