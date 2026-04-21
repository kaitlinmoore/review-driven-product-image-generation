# Methodology Decisions Log

A chronological-ish log of the decisions that shaped the project.
Each entry covers what was decided, what alternatives were considered,
and why the choice was made. Useful source for the
"methodology" section of the writeup.

## Product Selection

### Six products from at least 3 categories

The assignment requires three products from three different categories.
The project went beyond the minimum to six products spanning five
categories so we would have some wiggle room to shape the final project or 
just have a larger sample set if we want.

Decision criteria for individual products:
- Visual variety across the set (simple geometry → complex scenes)
- A mix of brand-mark visibility (logoed vs. unbranded)
- Sufficient review volume to filter aggressively

The decision document `exploration/candidates/` retains 40+ candidate
CSVs documenting which products were considered before the final six
were chosen.

### Headphones added late

The other five products were chosen at an early team meeting from a
shortlist (water_bottle, chess_set, jeans, backpack). Board game was dropped
after exploraitonr revealed poor review signal. Headphones were added later
to broaden Electronics representation and specifically to test how the pipeline
handles a product with a prominent text logo (Beats Studio3). This is documented
as an addendum in the candidate-selection thinking. The final canonical six are the
product set the analysis can be built on, either with a subset of 3 or all 6.

### Backpack as the limit case

The backpack's reviewer community focuses on durability and function
rather than appearance. Only 4% of raw reviews pass the visual-content
filter, leaving exactly 30 ranked reviews. Because the agent loop
skips the top 30 reviews (those are already in the initial prompt), the
backpack has zero reviews available for refinement. Rather than tune
the pipeline to accommodate, the decision was to **keep backpack in
the canonical six as a baseline case**. The output shows what the
pipeline produces with no review-driven refinement, which is itself an
informative comparison point.


## Models and APIs

### Two image-generation models for the Q3 comparison

For the canonical pipeline, two image models are compared:

- **FLUX.1-schnell** (open-weights, routed via fal-ai for inference
  speed) | represents the current state of open-weights diffusion.
- **gpt-image-1.5** (OpenAI, the successor to DALL-E 3) | represents
  current-state proprietary diffusion. DALL-E 3 itself is being retired
  May 2026, so gpt-image-1.5 is the appropriate proprietary choice.

Considered and not chosen: SDXL (older), Stable Diffusion 3 (less
adoption), Midjourney (no public API). The two chosen represent
"open-weights leading edge" and "proprietary leading edge". Clean
two-axis comparison.

### Mistral-7B-Instruct-v0.3 for the inner loop

The notebook used Mistral-7B-Instruct-v0.3 for prompt refinement
(`stepPrompt`) and self-rating (`ratePrompt`). Kept for the canonical
pipeline because:
- Open-weights (graders can reproduce)
- Quantizes to 4-bit cleanly via bitsandbytes (~5 GB VRAM, fits on
  consumer GPUs)
- Strong instruction-following at that size
- Already validated by the notebook prototype

Considered and not chosen: Llama 3 (similar capabilities, no
particular advantage), gpt-4o-mini for the inner loop (would have
worked but introduces external API dependency for what should be a
local refinement signal).

### gpt-4o-mini for the visual-content filter

For per-review classification (decides whether a review is "visually
descriptive"), the filter uses gpt-4o-mini with strict JSON output.
Considered and rejected: a vocabulary-based regex filter
(`exploration/visual_vocabulary.py` and
`exploration/compare_gating_strategies.py` exist as the methodology
artifact for this comparison). The vocabulary filter was simpler and
free, but gave too many false positives (matched "red" in "the box
arrived in the red of dawn") and false negatives (missed reviews that
described appearance without using common color words).

The chess_set product was used as the test bed for this comparison;
the comparison output is in `exploration/` though the specific
artifacts have been cleaned out of the main `data/` tree.

### gpt-4o (full model) for prompt and feature work

For initial-prompt generation and structured-feature extraction,
gpt-4o (not the mini version) is used. Reasons:
- Both tasks benefit substantially from gpt-4o's stronger writing and
  visual reasoning
- Volume is low (one initial-prompt call per product, a handful of
  structured-feature calls per product) so cost is manageable
- gpt-4o vision is needed for the image-extraction sources
  (`ground_truth`, `generated`)

## Experimental Design

### "No ground-truth leakage" boundary

Currently: **ground-truth images and ground-truth-derived features
are used ONLY for evaluation, NEVER as a feedback signal inside the generation loop**.

Rationale: The research question: "given only what customers said
about a product, how closely can a generative pipeline reconstruct
what the product looks like?" only holds if ground truth never
enters the generation loop. If it did, the experiment would collapse
to a denoising-against-reference problem, which is a fundamentally
easier (and different) question.

This boundary is documented in `REPRODUCIBILITY.md` under
"Experimental Design — No Ground-Truth Leakage" with a per-script
table showing which scripts read ground-truth artifacts. Only one
script does (`extract_structured_features.py --source ground_truth`),
and its output never feeds back into generation.

**Status**: Course-staff guidance is pending on whether
ground-truth-informed refinement might be allowed as an additional
config. The pipeline is structured so that an opt-in variant could
be added without rearchitecting, but the current canonical run holds
the boundary strictly.

### Three quality-signal configurations as ablations

Three ablation configurations are run, each isolating one variable:

| Config | Quality signal | Reference |
|---|---|---|
| **v1_title_clip** | CLIP cosine | Product title (from metadata) |
| **v2_initial_prompt_clip** | CLIP cosine | Contents of `initial_prompt.txt` |
| **v3_initial_prompt_features** | Structured-feature agreement | Pre-extracted features from initial prompt |

The **v1 → v2** comparison isolates the **reference text**: same
metric (CLIP cosine), different reference (product title vs. the
~250-word visual description). The hypothesis: a richer reference
should give CLIP a better target, producing stronger refinement.

The **v2 → v3** comparison isolates the **metric**: same reference
(initial prompt's visual content), different metric (CLIP cosine vs.
per-field structured agreement). The hypothesis: a structured metric
should give the loop more interpretable feedback than a single cosine.

Each step changes one variable. A reviewer asking "but which change
mattered?" gets a clean answer.

### Quality-target calibration per metric

The CLIP-cosine quality target is set to 0.5 (the value used in the
notebook's test invocation). Structured-feature agreement uses 0.7
because it operates on a different scale: CLIP cosines for product
photos top out around 0.4, while structured-feature agreement is
naturally in the 0.4–0.7 range for unrefined images. The two target
values are calibrated to mean "moderately challenging" on each
respective metric's natural scale.

This is not a per-config tuning knob (which would muddy the
ablation); it's a metric-scale adaptation that keeps both configs in
the same operating regime.

### Iter / threshold parameter values

The agent loop's iteration parameters use the values from the
notebook's test invocation:
- `iter_start = 10` (initial per-image refinement budget)
- `iter_max = 30` (ceiling on retuned iter budget)
- `iter_min = 1` (floor)
- `descriptiveness_threshold = 70` (Mistral self-rating target)

These give the inner loop enough room to generate diverse
(descriptiveness, quality) data points across multiple iterations per
image, which is what the surrogate-projection retune needs to fit a
non-degenerate quadratic. Smaller values were tried initially and
produced degenerate behavior (loop ran 1 iter per image regardless of
parameters).

## Architectural Design

### Per-config artifacts plus canonical pointers

Multiple ablation configs need to coexist on disk without overwriting
each other. The artifact-naming scheme separates per-config files from
canonical pointers:

```
Per-config (always written):
  data/{slug}/converged_prompt_{model}_{config}.txt
  data/{slug}/generated_image_{model}_{config}.png
  data/{slug}/agent_run_{model}_{config}_meta.json

Canonical pointers (written by default; suppressed with --no-promote):
  data/{slug}/converged_prompt_{model}.txt
  data/{slug}/generated_image_{model}.png
  data/{slug}/converged_prompt.txt   (= FLUX-promoted; what extract reads)
```

Per-config files are the truth for each named run. Canonical pointers
are convenience aliases that downstream stages
(`extract_structured_features.py --source converged`, ad-hoc analysis)
can read without knowing which config is currently "blessed." Running
a second config with `--no-promote` preserves the first as canonical
while still capturing the second's artifacts.

### Replay cache for reproducibility

Every external API call and heavy local model inference is wrapped via
`replay.py`'s `cached_call`. Each call's inputs are hashed
deterministically; the resulting cache entry is keyed by that hash.
On subsequent runs with identical inputs, the cache hit returns the
prior output without calling live.

The cache supports three modes via the `REPLAY_MODE` environment
variable:
- `record` (default) | call live on cache miss, return cached on hit
- `replay` | return cached on hit, raise on miss (no live calls)
- `force_record` | always call live, overwrite cache

The committed replay cache lets graders reproduce the canonical
results offline by setting `REPLAY_MODE=replay` and re-running the
same scripts. No API keys needed, no GPU, no gated-model access.

System-prompt SHA256 hashes are baked into cache-key inputs so any
edit to a prompt automatically invalidates affected entries, meaning no
silent drift between code and cached results.

