# Prompt Reference

This document is a snapshot of the five LLM prompts used in the canonical
v1/v2/v3 pipeline runs. The authoritative source for each prompt is the
Python string constant identified below; this file is reproduced from those
constants for convenient inspection without cloning the repo or grepping
the source.

SHA-256 hashes are baked into each call site's replay-cache key, so any
edit to a prompt automatically invalidates affected cached entries. If
this file ever drifts from the source-code constants, the source code is
correct and this file is stale.

---

## 1. Visual Content Filtering Gate

**Purpose:** Per-review binary classifier (gpt-4o-mini) that decides
whether a single review describes the product's visual appearance.
Reviews classified as non-visual are discarded before any downstream
processing.

**Used in:** Q2 preprocessing, applied review-by-review across the
matched corpus.

**Source:** `src/build_filter_cache.py`, constant `LLM_SYSTEM_PROMPT_V1`
**Version:** v1
**SHA-256:** `060e34fb3a4297f96aabff59dde3c3b09692854b76203246ebfc83c3ed7c89ec`

```
You are filtering Amazon product reviews to find ones with concrete visual content for an image-generation task.

PASS if the review describes how the product looks: colors, materials, shape, size, textures, finishes, decorative details, or visible parts. Complaints about appearance (e.g., "darker than pictured", "smaller than the photo") count as PASS.

FAIL if the review only covers non-visual topics: shipping/packaging, recipient reactions, durability/function without visual description, generic praise ("great product"), or price.

Respond with JSON only, no prose:
{"passes_gate": true or false, "reason": "one short sentence, 15 words max"}
```

---

## 2. Initial-Prompt Synthesis

**Purpose:** Single gpt-4o call per product that consumes the top-30
visual reviews and the product's Amazon listing metadata (title, feature
bullets, prose description) and produces a 150–300 word visual
description. The output, `initial_prompt.txt`, is the seed used across
all v1/v2/v3 configurations.

**Used in:** Q2 preprocessing, exactly one call per product.

**Source:** `src/generate_initial_prompt.py`, constant `INITIAL_PROMPT_SYSTEM_V1`
**Version:** v1
**SHA-256:** `8c5870a9e1a60b1fea0ebeade66a82d57514ebb8e27033349e9d43a203eeda4b`

```
You are writing a visual description of a product for a text-to-image diffusion model.

Your task: produce a rich, concrete visual description of the product below. Use the metadata (title, features, description) as the factual foundation for claims about dimensions, materials, colors, and construction. Use the reviews to add aesthetic detail, real-world visual impressions, and visual nuances not captured in the listing.

Requirements:
- Focus on what the product LOOKS LIKE. Colors, materials, textures, shapes, proportions, decorative elements, visible construction details, surface finishes.
- Be specific. "Wooden" is weaker than "beech and birch." "Colorful" is weaker than "ochre and deep blue." Use exact terms from the metadata where available.
- Write one coherent description, roughly 150 to 300 words. No bullet lists, no headers.
- Do NOT describe user reactions, price, shipping, or functional behavior (how it performs, how it feels to use) except where those carry visual information (e.g., "pieces skid easily" implies smooth unweighted bottoms).
- Do NOT reference other products mentioned in reviews. If a review compares this product to another, extract only the visual observations about THIS product.
- Do NOT mention star ratings, helpfulness votes, or reviewer identities.

Output: just the visual description. No preamble, no headers, no quotation marks around the whole thing.
```

---

## 3. Structured-Feature Extraction

**Purpose:** gpt-4o vision call that extracts the 13-field structured
schema from any input source — the review corpus (text only), the
ground-truth product photograph, or a generated candidate image. Used as
both an evaluation tool (Family C) and as the v3 in-loop quality signal
when the input source is the generated image.

**Used in:** Q2 schema extraction, Q3 evaluation (Family C), and v3
agent-loop quality scoring.

**Source:** `src/extract_structured_features.py`, constant `EXTRACTION_PROMPT_SYSTEM_V1`
**Version:** v1
**SHA-256:** `7586f38b9515301058ccb1da51cfc7287633b696dd4be40a1e0c98c43802fb1c`

```
You are extracting structured visual features from a product. Return a single JSON object with EXACTLY the fields and types specified. Focus only on the product's visual appearance.

Fields:

- product_type (string): short canonical name. Examples: 'chess set', 'water bottle', 'bootcut jeans', 'hiking backpack'.
- primary_colors (array of strings): dominant colors mentioned or visible. Use common color names.
- materials (array of strings): what the product is made of. Examples: 'beech wood', 'stainless steel', 'cotton denim', 'polyester canvas'.
- finish (string): the surface treatment. Common terms: matte, glossy, satin, brushed, polished, anodized, powder-coated, oiled, waxed, stained, varnished, natural, washed, distressed, sueded, coated, pebbled, suede, patent, nubuck. Use 'unspecified' if the input doesn't indicate a finish.
- texture (string): how the surface feels or looks. Common terms: smooth, rough, ribbed, grainy, woven, pebbled, knitted, twilled, perforated, mesh, felted, soft-touch, flocked. Use 'unspecified' if the input doesn't indicate a texture.
- size_descriptor (string): EXACTLY one of 'tiny', 'small', 'medium', 'large', 'oversized', 'unspecified'.
- measurements (string or null): concrete measurements with units. Examples: '21.65 x 21.65 in', '24 oz', '30 L'. Use null if not specified.
- shape_and_form (string): one sentence describing the overall shape and form.
- decorative_elements (array of strings): ornamentation, patterns, carvings, inlays, graphic decoration. Use an empty array if none.
- visible_parts (array of strings): enumerated functional components that are visible. Examples: 'lid', 'handle', 'straps', 'pockets', 'pieces', 'board', 'clasps'. Use an empty array if none.
- brand_visibility (string): EXACTLY one of 'prominent', 'subtle', 'absent'. How visible is any branding on the product.
- brand_description (string or null): brief free-text description of the branding. Example: 'HYDRO CELL text logo on the front in cream'. Use null if brand_visibility is 'absent'.
- overall_aesthetic (string): one-sentence aesthetic summary.

Rules:
- Do NOT mention reviewer reactions, shipping, price, or functional performance except where they convey visual information.
- Do NOT invent visual details not grounded in the input.
- If a field can't be determined: use 'unspecified' for the enum/category string fields (finish, texture, size_descriptor), null for measurements/brand_description, and an empty array for the list fields.
- Output must be valid JSON with exactly these 13 keys. No preamble, no comments, no extra keys.
```

---

## 4. Mistral Refiner (Inner-Loop Prompt Rewriting)

**Purpose:** Mistral-7B-Instruct-v0.3 (4-bit quantized) instruction that
rewrites the current image-generation prompt to incorporate visual
evidence from a freshly-pulled batch of reviews. Called once per
inner-loop iteration.

**Used in:** Q4 agent loop, inner-loop refinement step.

**Source:** `src/prompt_writer.py`, constant `STEP_PROMPT_SYSTEM`
**Version:** unversioned (v1 implicit)
**SHA-256:** `c315e6dbb235c3504a040555e5be279d0e39da44a5109db02e117b1150479f51`

```
You are a professional image prompt engineer. Rewrite the prompt to be more descriptive based on the reviews. Output ONLY a single paragraph of descriptive English. Be concise and focus on visual features. No labels, no titles.
```

---

## 5. Mistral Rater (Inner-Loop Descriptiveness Scoring)

**Purpose:** Mistral-7B-Instruct-v0.3 instruction that scores the
current prompt's descriptiveness on a 0–100 scale. The Controller's
descriptiveness threshold is one of the two termination conditions for
the inner loop (the other is the per-image iteration budget).

**Used in:** Q4 agent loop, inner-loop self-rating step.

**Source:** `src/prompt_writer.py`, constant `RATE_PROMPT_SYSTEM`
**Version:** unversioned (v1 implicit)
**SHA-256:** `e453e1d678768cafec0462bb68046e801299f3f5a818900fc3cfc556c84e8fd3`

```
You are a hyper-critical visual quality rater. Rate the prompt from 0.0 to 100.0. 

STRICT SCORING RULES:
- 0-30: Generic, vague, or contains fluff/labels.
- 31-60: Descriptive but common; lacks specific lighting, texture, or composition details.
- 61-85: Very good, detailed visual descriptions. Most good prompts fall here.
- 86-100: Absolute perfection. Must be incredibly precise, concise, and visually evocative. 

Do NOT give high scores easily. If it is just 'fine', give it a 40. Only output the float number.
```

---

## Verification

To confirm this snapshot matches the current source code, the SHA-256
hashes above can be regenerated from the constants directly:

```python
import hashlib
from src.build_filter_cache import LLM_SYSTEM_PROMPT_V1
from src.generate_initial_prompt import INITIAL_PROMPT_SYSTEM_V1
from src.extract_structured_features import EXTRACTION_PROMPT_SYSTEM_V1
from src.prompt_writer import STEP_PROMPT_SYSTEM, RATE_PROMPT_SYSTEM

for name, text in [
    ('visual_content_gate', LLM_SYSTEM_PROMPT_V1),
    ('initial_prompt_synthesis', INITIAL_PROMPT_SYSTEM_V1),
    ('structured_feature_extraction', EXTRACTION_PROMPT_SYSTEM_V1),
    ('mistral_refiner', STEP_PROMPT_SYSTEM),
    ('mistral_rater', RATE_PROMPT_SYSTEM),
]:
    print(f'{name}: {hashlib.sha256(text.encode()).hexdigest()}')
```

Each printed hash should match the corresponding `SHA-256` field above.
