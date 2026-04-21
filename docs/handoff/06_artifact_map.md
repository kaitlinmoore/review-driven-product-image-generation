# Artifact Map

For each report section and rubric component, where to find the
evidence on disk. Use as a lookup reference while drafting — open
this doc next to the writeup and pull file paths directly.

The numerical results referenced throughout (CLIP / DINOv2 / SigLIP
image-vs-image cosines and structured-feature agreement scores) are
in `eval_results/summary.csv` (one row per (product, model, config)
combination) and `eval_results/per_product/{slug}.json` (per-product
detail with per-field breakdowns).

## Q1 | Product Selection and Data Collection

What evidence exists, and what to cite for each part of Q1.

### Selection rationale

| What you need | Where to find it |
|---|---|
| Why these 6 products | `docs/handoff/03_data_processing.md` § "Selection Criteria for the Canonical Six" — write-ready paragraphs |
| Why this approach (visual variety, popularity, brand visibility) | `docs/handoff/05_decisions_log.md` § "Six products from at least 3 categories" |
| The candidate pool considered before final selection | `exploration/candidates/*.csv` — 40+ candidate CSVs by category × popularity-band |
| The product-selection methodology (search strategy) | `exploration/explore_candidates.py` |
| Why backpack despite low review signal | `docs/handoff/05_decisions_log.md` § "Backpack as the limit case" |
| Why headphones added late | `docs/handoff/05_decisions_log.md` § "Headphones added late" |

### Per-product collected data

| What you need | Where to find it |
|---|---|
| Product titles, descriptions, features | `data/{product}/metadata.json` |
| Raw customer reviews | `data/{product}/reviews.jsonl` (gitignored — large; reproducible via `pull_product_data.py`) |
| Reference product photos | `data/{product}/images/` (mix of main view and alternate views) |
| Quick stats (review counts, rating averages) | `data/{product}/summary.json` |
| The data-acquisition mechanism | `src/pull_product_data.py` (streams gzipped JSONL over HTTP from McAuley Lab Amazon Reviews 2023) |

### Per-product reference table for Q1 writeup

```
backpack         Sports & Outdoors    Venture Pal 40L Hiking Backpack    parent_asin: B08JD12G4X
chess_set        Toys & Games         Chess Armory 15 Inch Wooden        parent_asin: B09JZNWFFG
espresso_machine Home & Kitchen       Breville Barista Express BES870XL  parent_asin: B0B3D2KYNS
headphones       Electronics          Beats Studio3 over-ear             parent_asin (see metadata.json)
jeans            Clothing/Shoes/Jewl  Levi's Women's Mile High Skinny    parent_asin: B08KR19G7S
water_bottle     Home & Kitchen       HYDRO CELL Stainless Steel         parent_asin: B09L7RZ96X
```

## Q2 | LLM Analysis of Customer Reviews

The four LLM-driven analysis stages and their outputs.

### Stage outputs

| Stage | Code | Per-product output |
|---|---|---|
| 1. Visual-content filter (per-review) | `src/build_filter_cache.py` | `data/filter_caches/{product}_filter_decisions_v1.json` |
| 2. Review ranking + dedup | `src/preprocess_reviews.py` | `data/{product}/reviews_ranked.jsonl` |
| 2. Prompt-context assembly | `src/build_prompt_context.py` | `data/{product}/prompt_context.txt` |
| 3. Initial prompt distillation | `src/generate_initial_prompt.py` | `data/{product}/initial_prompt.txt` (+ `_meta.json` provenance) |
| 4. Structured-feature extraction | `src/extract_structured_features.py` | `data/{product}/structured_features_*_v1.json` (+ `_meta.json` per source) |

### System prompts (the actual prompt-engineering)

The system prompts ARE the analysis design. Each is versioned and
SHA-hashed into cache keys. For the writeup, cite the prompt text
verbatim from the source files:

| Prompt | File | Constant name |
|---|---|---|
| Visual-content filter | `src/build_filter_cache.py` | `FILTER_PROMPT_SYSTEM_V1` |
| Initial-prompt distillation | `src/generate_initial_prompt.py` | `INITIAL_PROMPT_SYSTEM_V1` |
| Structured-feature extraction | `src/extract_structured_features.py` | `EXTRACTION_PROMPT_SYSTEM_V1` |

### Per-source structured-feature comparison data

For each product, the same 13-field schema extracted from five
different reference inputs:

| Source | What it tells you |
|---|---|
| `structured_features_initial_v1.json` | What the initial prompt described (the LLM-distilled visual description) |
| `structured_features_prompt_context_v1.json` | What the LLM saw before distillation (metadata + top reviews combined) |
| `structured_features_metadata_only_v1.json` | What metadata alone contains (no review signal) |
| `structured_features_reviews_only_v1.json` | What top reviews alone contain (no metadata signal) |
| `structured_features_ground_truth_v1.json` | What the actual product looks like (reference) |

For the Q2 analytical narrative, compare these per product to show
what each input source contributes (e.g., metadata captures specs;
reviews capture finish/color nuance).

### Filter pass rates by product (Q2 writeup table)

Pull from `data/filter_caches/*.json` (counted via aggregation).
Pre-computed reference table in `docs/handoff/03_data_processing.md`
§ "Filter Pass Rates by Product".

## Q3 | Image Generation with Diffusion Model

Most of the analytical content lives here. Use the eval CSV for
numerical comparisons; per-product JSONs for narrative detail.

### Per-(product, model, config) generated artifacts

| What | Where (replace placeholders with actual values) |
|---|---|
| The refined prompt | `data/{product}/converged_prompt_{model}_{config}.txt` |
| The generated image | `data/{product}/generated_image_{model}_{config}.png` |
| In-loop trajectory provenance | `data/{product}/agent_run_{model}_{config}_meta.json` |
| Structured features extracted from generated image | `data/{product}/structured_features_generated_{model}_{config}_v1.json` |
| Structured features extracted from converged prompt | `data/{product}/structured_features_converged_{model}_{config}_v1.json` |

`{model}` is `flux` or `gpt`. `{config}` is one of:
- `v1_title_clip` (CLIP cosine vs product title — baseline)
- `v2_initial_prompt_clip` (CLIP cosine vs initial_prompt.txt content)
- `v3_initial_prompt_features` (structured-feature agreement vs initial features)

### Numerical comparison evidence (the eval results CSV)

`eval_results/summary.csv` — 36 rows × 15 columns. One row per
(product, model, config) combination. Columns:

| Column | Source | Meaning |
|---|---|---|
| `product`, `model`, `config` | Identifiers | The combination row |
| `in_loop_best_q` | `agent_run_*_meta.json` `qualities[best_idx]` | The quality score for the canonical image (using whichever signal that config used) |
| `in_loop_best_idx` | `agent_run_*_meta.json` `best_idx` | Which of the 3 image attempts was canonical (0/1/2) |
| `clip_img_vs_gt_mean`, `_max` | Computed | CLIP image-vs-image cosine, generated vs each ground-truth photo, aggregated |
| `dinov2_vs_gt_mean`, `_max` | Computed | DINOv2 cosine, same aggregation |
| `siglip_vs_gt_mean`, `_max` | Computed | SigLIP cosine, same aggregation |
| `gen_features_vs_initial` | Computed | Per-field agreement between generated image's features and initial-prompt features (preservation check) |
| `gen_features_vs_ground_truth` | Computed | Per-field agreement between generated image's features and ground-truth features (accuracy check, the most interpretable single number for Q3) |
| `conv_features_vs_initial` | Computed | Per-field agreement between converged prompt's features and initial-prompt features |
| `conv_features_vs_ground_truth` | Computed | Per-field agreement between converged prompt's features and ground-truth features |

### Narrative detail (per-product JSONs)

`eval_results/per_product/{slug}.json` — for each product, the same
metrics broken down by:
- Per-image scores for Family B (which ground-truth view did the
  generated image match best)
- **Per-field scores for Family C** (the failure-mode taxonomy: which
  of the 13 fields the pipeline reliably got right vs missed)

For "the pipeline got 8 of 13 fields right on chess_set under flux
v3" — pull from
`per_product/chess_set.json` → `configs.flux_v3_initial_prompt_features.gen_per_field_vs_ground_truth`.

### Image-model comparison (FLUX vs gpt-image)

Use `summary.csv`. For each (product, config), compare flux row vs
gpt row on `gen_features_vs_ground_truth` (the most interpretable
metric). Writeup-ready preliminary findings already in
`docs/handoff/04_evaluation_plan.md` and the eval-output discussion
in this session's history.

### Quality-signal ablation (v1 → v2 → v3)

For each (product, model), compare three rows of `summary.csv`. The
**v1 → v2** delta isolates the reference text. The **v2 → v3** delta
isolates the metric. Use Family C against ground_truth as the
INDEPENDENT verifier so neither config is graded by its own internal
metric.

### Side-by-side image comparison (for slides)

For each product, the writeup / slide deck should show:
- Reference photo from `data/{product}/images/main.jpg` (or the most
  representative non-excluded image)
- Generated image from `data/{product}/generated_image_{model}_{config}.png`
  for the BEST (model, config) combination per Family C
- Optionally, generated images from worst combinations to illustrate
  failure modes

## Q4 | AI Agentic Workflow

Architecture and reproducibility evidence.

### Architecture

| What | Where |
|---|---|
| Outer loop (image attempts + retune between) | `src/agent_loop.py` `agentLoop` function |
| Inner loop (Mistral refinement + self-rating) | `src/prompt_writer.py` `PromptWriter.improvementLoop` |
| Driver (iterates products, calls agentLoop, persists artifacts) | `src/run_agent_pipeline.py` |
| Quality-signal closure construction | `src/run_agent_pipeline.py` `build_quality_signal_fn` |
| Reproducibility wrapper | `src/replay.py` `cached_call` |
| Pipeline diagram | `docs/handoff/pipeline_diagram.png` (also Mermaid source in `02_pipeline_overview.md`) |

### Surrogate-projected adaptive iteration control

The novel agentic mechanism. The math is described in Jason's
whitepaper PDF (separate file). The implementation is in
`src/agent_loop.py`:

| Implementation piece | Where |
|---|---|
| Quadratic surrogate fit | `agent_loop.fit_quadratic` |
| Inverse projection (solve for x at target y) | `agent_loop.solve_for_x` |
| Projection onto feasible set | `agent_loop.pick_valid_solution` |
| Cumulative iteration update with clip | `agentLoop` retune block (after each image attempt) |

For the writeup, cite the per-product `iters_taken_per_image` and
`descriptivenesses` arrays from `agent_run_*_meta.json` to show the
adaptive behavior in practice (e.g., "image 0 ran 10 refinement
iters; after the retune, image 1 ran 2 iters; the agent allocated
proportionally to observed quality gain").

### Reproducibility

| Claim | Evidence |
|---|---|
| Replay mode reproduces canonical results offline | `REPRODUCIBILITY.md` § "Two Operating Modes" + the committed `replay_cache/` directory |
| Cache keys are content-addressed | `src/replay.py` `_cache_key` (SHA256 of inputs dict) |
| System-prompt edits invalidate stale cache | SHA256 of system prompts in `inputs` dict at every wrapped call site |
| All 11 production call sites are wrapped | `REPRODUCIBILITY.md` § "What Gets Cached" — 11-row table |
| No-leakage boundary respected by code | `REPRODUCIBILITY.md` § "Experimental Design — No Ground-Truth Leakage" — per-script audit table |

## Rubric Cross-Reference

The Q1–Q4 weights total 25%. The rubric (90% of grade) cuts across.
Where evidence for each rubric component lives:

### Question Understanding (10%)

Evidence: each Q section above demonstrates faithful coverage of what
the assignment asked for.

### Experiment Design (20%)

Where to point reviewers:

| Going beyond minimum | Evidence |
|---|---|
| 6 products instead of required 3 | `data/` directory listing + Q1 rationale |
| 2 leading-edge image models (FLUX + gpt-image), not "any two" | `src/gen_image_flux.py`, `src/gen_image_gpt.py` |
| 3 ablation configs testing different prompt-engineering signals | `eval_results/summary.csv` v1/v2/v3 rows; `docs/handoff/05_decisions_log.md` § "Three quality-signal configurations as ablations" |
| 13-field structured schema enabling cross-source comparison | `src/extract_structured_features.py` `STRUCTURED_FEATURES_SCHEMA_V1` |
| Surrogate-projected adaptive iteration control (novel mechanism) | `src/agent_loop.py` + Jason's whitepaper PDF |

### Analytics (30% — biggest single component)

Where the analytical depth lives:

| Type of analysis | Evidence |
|---|---|
| Per-product per-config trajectory analysis | `agent_run_*_meta.json` `qualities`, `descriptivenesses`, `iters_taken_per_image`, `best_idx` |
| Refinement-helps-vs-hurts pattern (best_idx distribution) | `eval_results/summary.csv` `in_loop_best_idx` column |
| The CLIP-vs-features divergence finding (CLIP says similar; features say different) | `eval_results/summary.csv` row for backpack flux: `clip_img_vs_gt_mean=0.81, gen_features_vs_ground_truth=0.16` |
| Why the backpack baseline | `docs/handoff/05_decisions_log.md` + `docs/handoff/03_data_processing.md` |
| Per-config v1/v2/v3 comparison with independent verification | `eval_results/summary.csv` ground-truth columns |

### Insights (30% — biggest single component)

Specific findings to report (with file paths to back them):

| Insight | Backing data |
|---|---|
| Mean structured-feature agreement against ground truth across 5 ablation-contributing products: ~0.78 | `eval_results/summary.csv` `gen_features_vs_ground_truth` column |
| Image-model winners vary by product (gpt better on backpack/water_bottle; flux better on chess_set v3, headphones, jeans) | Per-product comparison in `eval_results/summary.csv` |
| v3 (structured-feature signal) helps for some products, hurts for others (chess_set +0.08; jeans -0.20 vs v1) | `eval_results/summary.csv` |
| CLIP cosine alone can be misleading — the structured-feature comparison catches what CLIP misses | backpack flux: CLIP 0.81, features 0.16 |
| Per-field reliability: which fields the pipeline reliably gets vs misses | `eval_results/per_product/{slug}.json` `gen_per_field_vs_ground_truth` |
| Review-data quality varies dramatically by category (4% to 33% pass rate) | `data/filter_caches/*.json` aggregations; pre-computed table in `03_data_processing.md` |
| Mistral cycle cache hits cause v1 and v2 to produce identical images for many products | `eval_results/summary.csv` — many v1/v2 rows have identical metric values |

### Scientific Rigor (10%)

| What | Where |
|---|---|
| Code quality and module separation | `src/` directory — single-responsibility modules |
| Documentation | `README.md`, `REPRODUCIBILITY.md`, `docs/handoff/` |
| Reproducibility | Replay cache + REPRODUCIBILITY.md instructions |
| Provenance trails on every artifact | `*_meta.json` sidecars throughout `data/` |
| Versioned prompts with SHA invalidation | System-prompt SHAs baked into cache keys |

## Quick Reference Index

For each common writeup question, the single most useful starting file:

| "I need..." | Open this first |
|---|---|
| Product selection rationale | `03_data_processing.md` § "Selection Criteria" |
| Filter pass rates per product | `03_data_processing.md` § "Filter Pass Rates" |
| Initial prompt example for a product | `data/{product}/initial_prompt.txt` |
| Generated image for a (product, model, config) | `data/{product}/generated_image_{model}_{config}.png` |
| Numeric scores for any comparison | `eval_results/summary.csv` |
| Per-field breakdown for failure-mode analysis | `eval_results/per_product/{slug}.json` |
| In-loop trajectory for a (product, model, config) | `data/{product}/agent_run_{model}_{config}_meta.json` |
| Why a design decision was made | `05_decisions_log.md` |
| Pipeline architecture diagram | `pipeline_diagram.png` |
| Reproducibility instructions for graders | `REPRODUCIBILITY.md` |
