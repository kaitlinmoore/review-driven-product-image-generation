# Evaluation Plan

The framework for what gets compared, against what, using which metrics,
and what each comparison answers. Read this before drafting the
evaluation section of the writeup or building the comparison slides.

## Assignment Structure

The assignment's four questions and their grade weights:

| Question | Topic | Weight |
|---|---|---|
| Q1 | Product selection and customer review data collection | 5% |
| Q2 | Analysis of customer reviews with LLM | 7.5% |
| Q3 | Image generation with diffusion model | 7.5% |
| Q4 | Build an AI Agentic workflow | 5% |

Plus a separate rubric covering scientific rigor and analytical depth:

| Rubric component | Weight |
|---|---|
| Question understanding | 10% |
| Experiment Design (creativity, diversity, novelty) | 20% |
| Analytics (clear rationale for what worked / didn't) | 30% |
| Insights (meaningful, informative findings) | 30% |
| Scientific Rigor (code, documentation, reproducibility) | 10% |

The Q1/Q2/Q3/Q4 weights total 25% of assignment grade; the rubric
totals 100% of an evaluative dimension applied across the whole
project. The bulk of grading attention goes to **analytics** and
**insights** (60% combined). Plan accordingly: don't just describe
what happened, explain WHY and what we learned.

## Q1 | Product Selection and Data Collection (5%)

**What the assignment asks:**
- Select 3 products from different categories
- Consider categories and popularity levels
- Explain selection rationale
- Collect product descriptions and customer reviews

**What we delivered:**
- 6 products (2× the requirement) spanning 5 categories: Home & Kitchen
  (water_bottle, espresso_machine), Toys & Games (chess_set),
  Clothing (jeans), Sports & Outdoors (backpack), Electronics (headphones)
- Selection methodology backed by streaming the McAuley Lab Amazon
  Reviews 2023 dataset over HTTP and producing 40+ candidate CSVs
  (`exploration/candidates/`)
- Selection rationale spans visual variety (simple → complex), brand
  visibility (logoed vs unbranded), and review-pool richness (varies
  from 4% to 33% visually-relevant after filtering)

**Artifacts to cite in the writeup:**
- `data/{product}/metadata.json` — title, description, features per product
- `data/{product}/reviews.jsonl` — raw review records (gitignored;
  reproducible via `pull_product_data.py`)
- `data/{product}/images/` — reference photos
- `data/{product}/summary.json` — counts and quick-look info
- `exploration/candidates/*.csv` — candidate pool considered for selection
- `docs/handoff/03_data_processing.md` § "Product Selection" — write-ready
  rationale text

**Bonus narrative:** the backpack edge case (only 30 visually-rich
reviews after filtering) shows that product selection has consequences
downstream — not all products generate enough useful review signal to
support iterative refinement. That observation justifies including
backpack as a baseline rather than excluding it. See
`docs/handoff/05_decisions_log.md` § "Backpack as the limit case."

## Q2 | LLM Analysis of Customer Reviews (7.5%)

**What the assignment asks:**
- Use LLM API to extract valuable information from review text
- Examples: summarization, feature extraction, topic extraction,
  sentiment analysis
- Use prompt engineering, RAG, or combination
- Consider chunking strategies given API token limits
- Could use vector database for embeddings
- Effective output should feed the diffusion model

**What we delivered:**

A four-stage LLM-driven analysis pipeline, each stage using prompt
engineering with strict-JSON outputs to extract increasingly
structured information:

1. **Per-review visual-content filter** (`build_filter_cache.py`) —
   gpt-4o-mini classifies each of ~5,000 reviews as visually
   descriptive or not, with a brief reason string. Strict JSON-mode
   output; system prompt SHA hashed into cache keys for reproducibility.
2. **Review ranking and prompt-context assembly**
   (`preprocess_reviews.py` → `build_prompt_context.py`) — dedup by
   `(user_id, timestamp)`, rank by `helpful_vote`, take top 30 visually
   descriptive reviews per product, format with metadata into
   `prompt_context.txt`.
3. **Initial-prompt distillation** (`generate_initial_prompt.py`) —
   single gpt-4o call per product, system prompt instructs the model
   to write a 150–300 word visual description suitable for diffusion.
4. **Structured-feature extraction** (`extract_structured_features.py`)
   — gpt-4o (and gpt-4o vision for image sources) extracts a 13-field
   visual schema across seven different source types per product:
   `initial`, `prompt_context`, `metadata_only`, `reviews_only`,
   `ground_truth`, `converged` (per config), `generated` (per config).

The 13-field schema is the project's specific answer to "what is an
effective output that can feed the diffusion model" — it's the same
schema used to compare what the initial prompt described, what ground
truth shows, and what the generated image rendered.

**Artifacts to cite in the writeup:**
- `data/{product}/reviews_ranked.jsonl` — filtered + ranked
- `data/{product}/prompt_context.txt` — LLM input
- `data/{product}/initial_prompt.txt` — LLM-distilled visual description
- `data/{product}/structured_features_*_v1.json` — per-source 13-field outputs
- `data/filter_caches/{product}_filter_decisions_v1.json` — per-review classifications
- System prompts in `src/build_filter_cache.py`, `src/generate_initial_prompt.py`, `src/extract_structured_features.py` (each is a versioned, SHA-hashed prompt)

**Analytical depth opportunities:**
- **Filter pass rates** vary 4×: chess_set 33% pass, backpack 4% pass.
  Reflects how visually-focused each product's reviewer community is.
  Write up as a finding about review-data quality varying by category.
- **Comparison across the 5 reference structured-feature sources**
  (initial, prompt_context, metadata_only, reviews_only, ground_truth)
  shows what each input type contributes. E.g., `metadata_only`
  captures specs (measurements, materials) that reviews miss;
  `reviews_only` captures color and finish nuances that metadata
  misses. The `initial` prompt distills both. Per-field comparison
  shows where each source is rich vs sparse.
- **Why no RAG / vector database**: deliberate choice. The
  per-product review pool (≤2,000 entries) fits comfortably in the
  prompt context after filtering. RAG adds engineering complexity
  without analytical benefit at this scale. Document as a design
  choice in the writeup.

## Q3 | Image Generation with Diffusion Model (7.5%)

**What the assignment asks:**
- Craft prompts from the Q2 output
- Use at least 2 different image generation models
- Generate 3–5 images per product based on crafted prompts
- Iterate on prompts based on initial results
- Compare AI-generated vs actual product images: similar / different / in what dimensions / why
- Compare different image generation models: which is better, why
- Provide analyses and explanations

**What we delivered:**

Beyond the bare requirement: 3 prompt-engineering configurations
(v1/v2/v3) × 2 image models (FLUX.1-schnell, gpt-image-1.5) × 3
images per attempt × 6 products = **108 generated images** with full
provenance trace.

**Iteration:** the agent loop iteratively refines the prompt with
Mistral-7B between image attempts (1–10 inner-loop refinement steps
per image), then chooses the best-scoring image as canonical. This
satisfies "iterate on your prompts based on initial results."

**Comparison vs actual product images** — the most important Q3
analysis, computed from the structured-feature extractions and from
embedding-cosine metrics:

### Family B | Generated-vs-Ground-Truth Image Cosines

For each (product, model, config) canonical image, compare to the
ground-truth photos with three independent embedding models:

| Metric | Function | What it captures |
|---|---|---|
| CLIP image-vs-image | `eval_image.clip_image_similarity` | Holistic visual-semantic alignment |
| DINOv2 cosine | `eval_image.dinov2_similarity` | Object-level structural similarity |
| SigLIP cosine | `eval_image.siglip_similarity` | Alternative VLM perspective |

Three independent models guard against findings being an artifact of
any one embedding's biases. If the three agree directionally, the
conclusion is robust.

### Family C | Structured-Feature Agreement vs Ground Truth

For each (product, model, config), compare the per-config generated
features (`structured_features_generated_{model}_{config}_v1.json`)
to ground-truth features (`structured_features_ground_truth_v1.json`)
using `eval_image.feature_agreement`.

This is the most interpretable comparison: instead of "FLUX scored
0.69, gpt-image scored 0.71," you can say "FLUX got 8 of 13 fields
right; gpt-image got 9 of 13 — including correct brand text rendering
on headphones, which FLUX missed."

### Image-Model Comparison (FLUX vs gpt-image-1.5)

For each (product, config), compare FLUX's canonical image to
gpt-image's canonical image on Family B and Family C. Aggregate
across products to identify per-model strengths:
- Per-product table of best_q (in-loop) and ground-truth-comparison scores
- Per-model: which products did each model do best on?
- Per-field: which structured-feature fields does each model handle
  reliably (e.g., is gpt-image consistently better at brand text? is
  FLUX consistently better at fabric texture?)

### Quality-Signal Ablation (v1 → v2 → v3)

The three configurations compare different prompt-engineering
strategies:

| Config | Quality signal | Reference text/features |
|---|---|---|
| v1 | CLIP image-vs-text cosine | Product title (short) |
| v2 | CLIP image-vs-text cosine | initial_prompt.txt content (rich) |
| v3 | Structured-feature agreement | initial_prompt's 13-field features |

**v1 → v2** isolates the reference: same metric, different reference.
Hypothesis: a richer reference produces better refinement signal.

**v2 → v3** isolates the metric: same reference, different metric.
Hypothesis: structured-feature signal gives more interpretable
feedback than CLIP cosine.

For each comparison, the report should:
- Show per-product trajectories (`agent_run_*_meta.json` `qualities`,
  `descriptivenesses`, `iters_taken_per_image`, `best_idx`)
- Use Family B and/or Family C against ground_truth as INDEPENDENT
  evaluators (so neither config has the home-field advantage of being
  graded by its own metric)
- Report which config produces images closer to ground truth, NOT
  which config produces highest internal quality scores

**Artifacts to cite:**
- `data/{product}/converged_prompt_{model}_{config}.txt`
- `data/{product}/generated_image_{model}_{config}.png`
- `data/{product}/agent_run_{model}_{config}_meta.json` — full
  provenance with hyperparameters, trajectories, best_idx
- `data/{product}/structured_features_generated_{model}_{config}_v1.json`
- `data/{product}/structured_features_ground_truth_v1.json`
- `data/{product}/images/` — reference photos for visual side-by-side

**Analytical depth opportunities:**
- **The "best image isn't always the most refined" pattern** — across
  v1/v2/v3, `best_idx` is sometimes 0 (least-refined), sometimes 2
  (most-refined). This is a real finding about when iteration helps.
- **The per-field failure-mode taxonomy** — which structured-feature
  fields does the pipeline reliably get right vs miss? (e.g., colors
  are usually correct; brand text is systematically wrong; fine
  decorative elements are unreliable). This taxonomy is more
  interpretable than a single similarity number.
- **Model-specific strengths** — gpt-image vs FLUX likely differ on
  text rendering, photo realism, prompt adherence. Per-product
  comparison surfaces this.

## Q4 | AI Agentic Workflow (5%)

**What the assignment asks:**
- Design and build an AI Agentic workflow connecting all the above steps

**What we delivered:**

A genuine adaptive agent (not a fixed-pipeline script) implementing
**surrogate-projected adaptive iteration control**: after each
generated image, the loop fits a quadratic surrogate to observed
(descriptiveness, quality) and (cumulative iterations, quality) data,
and uses inverse projection to retune the descriptiveness threshold
and iteration budget for the next image attempt. The agent allocates
compute proportionally to task difficulty without prior knowledge of
that difficulty.

**Architecture:**
- Outer loop (`agent_loop.agentLoop`) — orchestrates per-image
  attempts and the surrogate-projection retune between them
- Inner loop (`PromptWriter.improvementLoop`) — refines the prompt
  with Mistral-7B, terminates on either descriptiveness threshold
  or iter cap (whichever fires first)
- Configurable quality signal (`build_quality_signal_fn`) — driver
  picks one of three signal/reference combinations per run, agent
  loop is signal-agnostic
- Replay-cached at every external call site so the entire workflow
  reproduces offline from cached artifacts

**Artifacts to cite:**
- `src/agent_loop.py` — orchestration logic with the quadratic-fit retune
- `src/prompt_writer.py` — Mistral wrapper and inner loop
- `src/run_agent_pipeline.py` — driver that iterates products and configs
- `src/replay.py` — reproducibility wrapper used by every call
- `docs/handoff/02_pipeline_overview.md` — Mermaid diagram of the workflow
- `docs/handoff/pipeline_diagram.png` — static rendering for slides

**Analytical depth opportunities:**
- **The retune mathematics** — describe the surrogate-projection
  approach (fit quadratic, inverse-project, project onto feasible
  set, clip to bounds). This is genuinely novel work and worth a
  short methodology subsection.
- **Parameter-regime sensitivity** — the design's adaptive behavior
  depends on parameter choices that allow the inner loop enough
  diversity to generate non-degenerate (descr, quality) data. The
  decisions log records what we learned about parameter regimes.
- **Replay reproducibility** — the offline-grader story.
  Cache-keyed-by-content-hash + system-prompt SHA validation makes
  the workflow methodologically rigorous in a way that ad-hoc
  pipelines aren't.

## Cross-Cutting Considerations

These apply across all four questions and should be brought up
wherever relevant in the writeup.

### The Backpack Baseline Case

backpack's review pool exhausts at the initial-prompt step (30
visually-rich reviews after filtering, all consumed by the initial
prompt). Its agent loop runs zero refinement iterations across all
configs. Treat as a **baseline showing the pipeline's lower bound**
without iterative refinement. Mention explicitly when reporting
averages so they aren't distorted. The other 5 products carry the
ablation analysis.

### No Ground-Truth Leakage

All three v1/v2/v3 configurations use only metadata- or
review-derived references for their quality signals. Ground-truth
artifacts are used ONLY for Family B and Family C post-hoc
evaluations. This boundary is what makes the research question
meaningful: "given only what customers SAID, how well can we
reconstruct what the product LOOKS like." If ground truth fed back
into generation, the experiment would be a denoising-against-reference
problem, which is a different question. Restate the boundary briefly
in the eval section so the reader understands the comparison is fair.

### Independent Verifiers

When ablating across configs that use different internal metrics
(notably v2 vs v3), use Family B and Family C scores against
ground_truth as the independent verifier. Otherwise the
"better" config is just the one whose internal metric is being
applied as the judge.

### Per-Field Reliability vs Aggregate Score

The structured-feature schema lets analysis go beyond a single
agreement number. Per-field breakdowns show WHICH aspects the
pipeline reliably gets right (often: product type, materials,
overall aesthetic) vs WHICH it reliably misses (often: brand text,
fine decorative elements, exact color terms). The per-field story
is more interpretable and more useful for the writeup's "insights"
than the single number.

## What's Computed vs What Still Needs Computing

**Already on disk after Phase 3 + Phase 4:**
- All in-loop quality trajectories (`agent_run_*_meta.json` per
  product/model/config)
- All structured-feature extractions for generated and converged
  sources per (product, model, config) — Phase 4 outputs

**Still to compute (post-hoc evaluation pass):**
- Family B image-vs-image scores: for each (product, model, config)
  generated image, compute CLIP / DINOv2 / SigLIP cosine against
  each non-excluded ground-truth image, then aggregate (mean is the
  natural choice)
- Family C feature-agreement scores comparing each generated and
  converged feature dict against:
  - `structured_features_initial_v1.json` (preservation check)
  - `structured_features_ground_truth_v1.json` (accuracy check)

The functions exist in `src/eval_image.py` (`clip_image_similarity`,
`dinov2_similarity`, `siglip_similarity`, `feature_agreement`). They
are cache-aware via `replay.py` so re-running is free. Producing the
post-hoc evaluation results is a small driver script (~100 LOC) that
iterates the (product, model, config) matrix, calls the appropriate
metric functions, and writes a summary CSV plus per-product detail
JSONs that the writeup and slide-deck team can pull from directly.

## Mapping to the Rubric

The four-question structure addresses the assignment prompts. The
rubric (90% of the grade) cuts across them. Here's how the project's
work maps to each rubric component:

**Question Understanding (10%)** — clear in each Q section above,
addressed by completing each Q's deliverables faithfully.

**Experiment Design (20%)** — creativity / diversity / novelty:
- 6 products instead of required 3 (2× the diversity)
- Two leading-edge image models (FLUX + gpt-image) instead of just any two
- Three structured ablation configs (v1/v2/v3) testing different
  prompt-engineering signals — most teams will run one configuration.
- 13-field structured schema enabling cross-source comparison —
  most teams will compare on holistic numeric metrics only
- Surrogate-projected adaptive iteration control — novel agentic
  mechanism vs typical fixed-budget loops

**Analytics (30%)** — clear rationale for what worked and didn't:
- Per-product variation across configs (best_idx distribution)
- Per-field reliability breakdown across products
- The "refinement helps sometimes, hurts other times" pattern
  examined per product and per quality signal
- The backpack case: a clean explanation of WHY refinement didn't
  help and what that means for the methodology

**Insights (30%)** — meaningful and informative findings:
- Per-product per-field failure-mode taxonomy
- Image-model comparative strengths
- The "better internal quality signal isn't necessarily better
  ground-truth alignment" finding (independent-verifier pattern)
- Review-data quality varies dramatically by category (4% to 33%
  visually-relevant pass rate)

**Scientific Rigor (10%)** — code, documentation, reproducibility:
- Replay cache and offline reproducibility
- System-prompt SHA hashing for cache invalidation
- Provenance sidecars on every generated artifact
- Repository documentation (README, REPRODUCIBILITY, handoff folder)
- Per-config artifact scheme that prevents ablation results from
  overwriting each other
