# Glossary

Plain-language definitions for the technical terms used in the project.
Reference document — look up terms as you encounter them in the other
handoff docs or in the writeup. Terms are grouped by topic; alphabetical
index at the bottom.

---

## Models and tools

**CLIP** — A vision-language model from OpenAI (specifically
`clip-ViT-B-32`). Maps both images and text into a shared embedding space
so they can be compared. Used in this project to measure how similar a
generated image is to a piece of reference text.

**CLIP cosine similarity** — A number between roughly 0 and 1 (in
practice 0.20–0.40 for product photos vs. their descriptions) that
measures how aligned an image is with a piece of text in CLIP's
embedding space. Higher means more aligned. Computed as the cosine of
the angle between the image embedding vector and the text embedding
vector.

**DINOv2** — A vision-only model from Meta (`facebook/dinov2-base`).
Produces image embeddings for image-vs-image comparison. Used in
post-hoc evaluation, not in the agent loop.

**FLUX.1-schnell** — A diffusion-based text-to-image model from Black
Forest Labs. One of the two image-generation models compared in this
project. Open-weights but routed via fal-ai (a remote inference
provider) for speed.

**gpt-image-1.5** — OpenAI's text-to-image model, the successor to
DALL-E 3 (which is being retired May 2026). The other of the two image
models compared.

**gpt-4o / gpt-4o-mini** — OpenAI's general-purpose language models.
gpt-4o is used for the initial-prompt generation and structured-feature
extraction (including vision). gpt-4o-mini is the cheaper variant used
for the visual-content filter gate that decides which reviews are useful.

**Mistral-7B-Instruct-v0.3** — An open-weights large language model from
Mistral AI. Used for the iterative prompt refinement (rewriting the
prompt) and self-rating (scoring the prompt's descriptiveness). Loaded
locally in 4-bit quantized form via the `bitsandbytes` library.

**SigLIP** — A vision-language model from Google
(`google/siglip-base-patch16-224`). Like CLIP but trained with a
different loss function. Used in post-hoc evaluation.

**Diffusion model** — A class of generative AI models that produces
images by progressively denoising random noise toward an image that
matches a text prompt. Both FLUX.1-schnell and gpt-image-1.5 are
diffusion models.

**VLM (vision-language model)** — Any model that can take an image as
input and produce text as output, or vice versa. CLIP, gpt-4o (when
used in vision mode), DINOv2, and SigLIP are all VLMs.

---

## Pipeline stages and concepts

**Initial prompt** — A roughly 250-word visual description of the
product, generated once per product by gpt-4o from the product's
metadata and top filtered reviews. The starting point for all
subsequent refinement and image generation.

**Iterative refinement (the agent loop)** — The process of repeatedly
rewriting the prompt using new batches of reviews, then generating
candidate images, then choosing the best image. The "agent" is the
control logic that decides how many refinement rounds to do per image
attempt.

**Refinement iteration** — One cycle of (Mistral rewrites the prompt
using a batch of reviews) + (Mistral self-rates the new prompt's
descriptiveness 0–100). Multiple iterations can happen per image
attempt.

**Descriptiveness** — Mistral's self-rating of how descriptive a
prompt is, on a 0–100 scale. Used inside the inner loop to decide
whether the prompt has been refined "enough." NOT a measure of how
good the generated image is — it only judges the text.

**Quality signal** — The metric used to score a generated image
against a reference. Three different quality signals are compared in
this project:
- **CLIP cosine vs. product title** (config v1)
- **CLIP cosine vs. initial_prompt.txt content** (config v2)
- **Structured-feature agreement vs. initial_prompt's pre-extracted
  features** (config v3)

**Quality target** — A numerical threshold the agent loop is trying to
reach with the quality signal. Used by the surrogate-projected
adaptive iteration control to decide how many refinement iterations
are still needed.

**Visual-content gate / filter gate** — A per-review classification
step where gpt-4o-mini decides whether a given review contains
information about the product's visual appearance (passes) or only
non-visual content like shipping or function (fails). Filters down
from raw reviews to "visually descriptive" reviews before assembling
the prompt context.

**Replay cache** — A directory (`replay_cache/`) of cached results
from every external API call and heavy local model inference. Allows
the entire pipeline to be re-run offline without API access by reading
cached outputs instead of calling live. Each cache entry is keyed by a
deterministic hash of the call's function name and inputs.

**Record mode / replay mode / force_record mode** — The three modes
the replay cache supports, controlled by the `REPLAY_MODE` environment
variable. Record (default) caches new calls and reuses cached ones;
replay reads cache only and errors on misses; force_record always
overwrites the cache with fresh live results.

---

## Evaluation and methodology

**Structured features** — A 13-field JSON schema describing a product's
visual properties: product type, primary colors, materials, finish,
texture, size descriptor, measurements, shape and form, decorative
elements, visible parts, brand visibility, brand description, and
overall aesthetic. Extracted from text descriptions or images using
gpt-4o, and used as a comparison structure across different sources
(initial prompt, ground-truth images, generated images, etc.).

**Ground truth** — The product's actual real-world appearance, as
captured by the reference photos pulled from Amazon. The "answer key"
the pipeline is being measured against. Available as both raw images
(`data/{product}/images/`) and as a structured-features dict
(`structured_features_ground_truth_v1.json`). **Crucially, ground
truth is used ONLY for evaluation — never as a feedback signal inside
the generation loop.** This boundary is what makes the experimental
question meaningful.

**Ablation** — A controlled comparison where exactly one variable is
changed between two experimental conditions, holding everything else
constant. Used to isolate which design choice produces an observed
effect. The v1 → v2 → v3 sequence in this project is structured as
two consecutive ablations:
- v1 → v2 changes only the reference text (title → initial_prompt)
- v2 → v3 changes only the metric (CLIP cosine → structured-feature
  agreement)

**Configuration / config name** — A named set of agent-loop parameters
(quality signal, reference, hyperparameters). Each run of the agent
pipeline is associated with a config name; per-config artifacts are
saved alongside canonical "blessed" pointers so multiple configs can
coexist on disk for ablation comparison.

**Surrogate-projected adaptive iteration control** — The mathematical
framework underlying the agent loop's adaptive behavior. After each
generated image, the loop fits a quadratic surrogate function through
the observed (descriptiveness, quality) and (cumulative-iterations,
quality) data points. It then projects what descriptiveness threshold
and iteration budget would be needed to reach the quality target, and
adjusts those parameters for the next image. Allows the agent to
allocate compute proportionally to task difficulty without prior
knowledge of how hard each task is.

**Agent loop / outer loop / inner loop** — The agent's nested control
structure. The outer loop generates one image per "image attempt" (3
attempts per product per model in the canonical configuration). The
inner loop refines the prompt with multiple Mistral rewrite cycles
within each image attempt. After each image attempt, the
surrogate-projection retune adjusts the inner loop's parameters for
the next image attempt.

---

## Alphabetical index

Ablation • Agent loop • CLIP • CLIP cosine similarity • Configuration •
Descriptiveness • Diffusion model • DINOv2 • FLUX.1-schnell • Filter
gate • gpt-4o • gpt-image-1.5 • Ground truth • Initial prompt • Inner
loop • Iterative refinement • Mistral-7B-Instruct-v0.3 • Outer loop •
Quality signal • Quality target • Record mode • Refinement iteration •
Replay cache • Replay mode • SigLIP • Structured features •
Surrogate-projected adaptive iteration control • Visual-content gate •
VLM
