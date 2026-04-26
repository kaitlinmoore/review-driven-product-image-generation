'''eval_image.py
-------------
Image similarity metrics for the generated-vs-ground-truth comparison stage,
and quality-signal evaluators usable inside the agent-loop refinement.

Three comparison modes are supported:

1. Image-vs-text (via CLIP)
   - compImage(image, reference_text) -> float
     Used in the agentLoop quality-signal closure for `--quality-signal clip_text`.
     The reference text is whatever the driver chose: product title (v1),
     initial_prompt.txt content (v2), etc.

2. Image-vs-image (via CLIP / DINOv2 / SigLIP)
   - clip_image_similarity(img_cand, img_gt) -> float
   - dinov2_similarity(img_cand, img_gt) -> float
   - siglip_similarity(img_cand, img_gt) -> float

3. Image-vs-structured-features (via gpt-4o vision + per-field agreement)
   - eval_structured_features(image, reference_features) -> float
     Used in the agentLoop quality-signal closure for `--quality-signal
     structured_features`. Extracts the 13-field structured feature dict from
     the candidate image and computes equal-weighted per-field agreement
     against a reference feature dict (loaded from a previously-extracted
     structured_features_*_v1.json).

Models are loaded lazily and cached module-wide so the first call to each
similarity function pays a one-time download/load cost. Subsequent calls are
cheap.
'''

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from replay import cached_call, image_hash


## MODEL IDS ##

ST_CLIP_NAME = 'clip-ViT-B-32'                         # sentence-transformers CLIP (image+text)
HF_CLIP_NAME = 'openai/clip-vit-base-patch32'          # transformers CLIP (image features)
DINOV2_NAME = 'facebook/dinov2-base'
SIGLIP_NAME = 'google/siglip-base-patch16-224'


## LAZY MODEL CACHES ##

_ST_CLIP = None          # sentence-transformers SentenceTransformer
_HF_CLIP = None          # tuple (processor, model) for transformers CLIP
_DINO = None             # tuple (processor, model)
_SIGLIP = None           # tuple (processor, model)


def _device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _get_st_clip() -> SentenceTransformer:
    '''Lazy-load the sentence-transformers CLIP for image-vs-text comparisons.'''
    global _ST_CLIP
    if _ST_CLIP is None:
        _ST_CLIP = SentenceTransformer(ST_CLIP_NAME)
    return _ST_CLIP


def _get_hf_clip():
    '''Lazy-load transformers CLIP (returns processor, model) for image-vs-image.'''
    global _HF_CLIP
    if _HF_CLIP is None:
        processor = CLIPProcessor.from_pretrained(HF_CLIP_NAME)
        model = CLIPModel.from_pretrained(HF_CLIP_NAME).to(_device()).eval()
        _HF_CLIP = (processor, model)
    return _HF_CLIP


def _get_dino():
    '''Lazy-load DINOv2 (returns processor, model).'''
    global _DINO
    if _DINO is None:
        processor = AutoImageProcessor.from_pretrained(DINOV2_NAME)
        model = AutoModel.from_pretrained(DINOV2_NAME).to(_device()).eval()
        _DINO = (processor, model)
    return _DINO


def _get_siglip():
    '''Lazy-load SigLIP (returns processor, model).'''
    global _SIGLIP
    if _SIGLIP is None:
        processor = AutoProcessor.from_pretrained(SIGLIP_NAME)
        model = AutoModel.from_pretrained(SIGLIP_NAME).to(_device()).eval()
        _SIGLIP = (processor, model)
    return _SIGLIP


## IMAGE-VS-TEXT (CLIP) | canonical flow ##

def compImage(image_input, ground_truth_text: str) -> float:
    '''CLIP cosine similarity between a generated image and a ground-truth
    text description (e.g. the product title).

    image_input: either a PIL.Image or a file path (str/Path).
    ground_truth_text: the string to compare against.

    Returns a float in [-1, 1] (practically almost always in [0, 1]).

    Replay-aware: cache key is (model, image pixel hash, text). Cached float
    is returned on hit without loading the CLIP model — replay-mode graders
    can skip the ~350 MB download entirely.'''
    inputs = {
        'metric': 'clip_text',
        'model': ST_CLIP_NAME,
        'image_hash': image_hash(image_input),
        'text': ground_truth_text,
    }

    def _live() -> float:
        model = _get_st_clip()

        if isinstance(image_input, (str, Path)):
            img_p = Path(image_input)
            if not img_p.exists():
                raise FileNotFoundError(f'Candidate image not found: {img_p}')
            img_cand = Image.open(img_p).convert('RGB')
        else:
            img_cand = image_input.convert('RGB')

        img_emb = model.encode(img_cand, convert_to_tensor=True)
        txt_emb = model.encode(ground_truth_text, convert_to_tensor=True)
        return util.cos_sim(img_emb, txt_emb).item()

    return cached_call('eval_clip_text', inputs, _live, format='json')


def evalImage(generated_image, ground_truth_text: str) -> float:
    '''Thin wrapper over compImage so the orchestration loop can import a
    single `evalImage` symbol. Kept for interface compatibility with the
    notebook's agentLoop.'''
    return compImage(generated_image, ground_truth_text)


## IMAGE-VS-IMAGE (CLIP / DINOv2 / SigLIP) ##
# These three functions mirror the REPRODUCIBILITY.md pinned-models list.

def _cosine_similarity_from_features(features: torch.Tensor) -> float:
    '''Cosine similarity between rows 0 and 1 of a (2, D) feature tensor.'''
    return F.cosine_similarity(features[0:1], features[1:2]).item()


def clip_image_similarity(img_cand_pil: Image.Image, img_gt_pil: Image.Image) -> float:
    '''Image-vs-image cosine similarity using transformers CLIP image features.
    Replay-aware: cached float returned on hit without loading CLIP.'''
    inputs = {
        'metric': 'clip_image',
        'model': HF_CLIP_NAME,
        'image_cand_hash': image_hash(img_cand_pil),
        'image_gt_hash': image_hash(img_gt_pil),
    }

    def _live() -> float:
        processor, model = _get_hf_clip()
        pt_inputs = processor(images=[img_cand_pil, img_gt_pil], return_tensors='pt').to(_device())
        with torch.no_grad():
            embeddings = model.get_image_features(**pt_inputs)  # (2, 512)
        return _cosine_similarity_from_features(embeddings)

    return cached_call('eval_clip_image', inputs, _live, format='json')


def dinov2_similarity(img_cand_pil: Image.Image, img_gt_pil: Image.Image) -> float:
    '''Image-vs-image cosine similarity using DINOv2 CLS-token embeddings.
    Replay-aware: cached float returned on hit without loading DINOv2.'''
    inputs = {
        'metric': 'dinov2',
        'model': DINOV2_NAME,
        'image_cand_hash': image_hash(img_cand_pil),
        'image_gt_hash': image_hash(img_gt_pil),
    }

    def _live() -> float:
        processor, model = _get_dino()
        pt_inputs = processor(images=[img_cand_pil, img_gt_pil], return_tensors='pt').to(_device())
        with torch.no_grad():
            outputs = model(**pt_inputs)
        # CLS token | Shape (2, hidden_dim)
        embeddings = outputs.last_hidden_state[:, 0]
        return _cosine_similarity_from_features(embeddings)

    return cached_call('eval_dinov2', inputs, _live, format='json')


def siglip_similarity(img_cand_pil: Image.Image, img_gt_pil: Image.Image) -> float:
    '''Image-vs-image cosine similarity using SigLIP image features.
    Replay-aware: cached float returned on hit without loading SigLIP.
    (Bug fix from cell 1's pseudocode: original reference to `processor` and
    `model` didn't call _get_siglip() to bind them.)'''
    inputs = {
        'metric': 'siglip',
        'model': SIGLIP_NAME,
        'image_cand_hash': image_hash(img_cand_pil),
        'image_gt_hash': image_hash(img_gt_pil),
    }

    def _live() -> float:
        processor, model = _get_siglip()
        pt_inputs = processor(images=[img_cand_pil, img_gt_pil], return_tensors='pt').to(_device())
        with torch.no_grad():
            embeddings = model.get_image_features(**pt_inputs)  # (2, 768)
        return _cosine_similarity_from_features(embeddings)

    return cached_call('eval_siglip', inputs, _live, format='json')


## STRUCTURED-FEATURE COMPARISON ##
# Used by the agent loop's `--quality-signal structured_features` mode.
# Per-field agreement against a reference feature dict.

# 13-field schema partitioned by comparison strategy. Equal-weighted across
# all 13 — every field contributes 1/13 to the final score.
_SET_FIELDS = ('primary_colors', 'materials', 'decorative_elements', 'visible_parts')
_EXACT_FIELDS = ('finish', 'texture', 'size_descriptor', 'brand_visibility', 'measurements')
_TOKEN_FIELDS = ('product_type', 'shape_and_form', 'brand_description', 'overall_aesthetic')
_ALL_FIELDS = _SET_FIELDS + _EXACT_FIELDS + _TOKEN_FIELDS  # 13 total


def _norm_str(x) -> str:
    '''Lowercase, strip; treat None and empty as the same canonical empty.'''
    if x is None:
        return ''
    return str(x).strip().lower()


def _set_jaccard(a, b) -> float:
    '''Set Jaccard for list-valued fields. Both empty -> 1.0 (agreement on
    absence). One empty, one not -> 0.0.'''
    set_a = {_norm_str(x) for x in (a or []) if _norm_str(x)}
    set_b = {_norm_str(x) for x in (b or []) if _norm_str(x)}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _exact_match(a, b) -> float:
    '''Case-insensitive exact-match for enum / short-string fields. Both
    None/empty -> 1.0; any other mismatch -> 0.0.'''
    na, nb = _norm_str(a), _norm_str(b)
    if na == nb:
        return 1.0
    return 0.0


def _token_jaccard(a, b) -> float:
    '''Token-set Jaccard for free-text fields. Splits on whitespace, lowercased,
    punctuation-stripped at token boundaries. Both empty -> 1.0; one empty,
    one not -> 0.0.

    Free-text fields can never reasonably match exactly across rephrasing, but
    semantically-similar descriptions share content tokens. This rewards
    overlap without being arbitrary about synonyms.'''
    na, nb = _norm_str(a), _norm_str(b)
    tokens_a = {t.strip(",.;:!?\"'()[]{}") for t in na.split() if t.strip(",.;:!?\"'()[]{}")}
    tokens_b = {t.strip(",.;:!?\"'()[]{}") for t in nb.split() if t.strip(",.;:!?\"'()[]{}")}
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def feature_agreement(features_a: dict, features_b: dict) -> float:
    '''Equal-weighted per-field agreement between two structured-feature dicts.

    Each of the 13 schema fields contributes 1/13 to the score.
    - Set-valued fields use set Jaccard.
    - Enum / short-string fields use exact-match (case-insensitive).
    - Free-text fields use token-set Jaccard.

    Returns a float in [0, 1]. Higher = closer agreement.'''
    return per_field_agreement(features_a, features_b)['overall']


def per_field_agreement(features_a: dict, features_b: dict) -> dict:
    '''Same scoring as `feature_agreement` but returns the per-field breakdown
    alongside the overall score. Useful for failure-mode analysis in the
    writeup ("which fields did the pipeline reliably get right; which did it
    miss") without recomputing.

    Returns:
        {
            'overall': float in [0, 1],
            'per_field': {field_name: float in [0, 1] for each of 13 fields},
        }'''
    per_field: dict[str, float] = {}
    for field in _SET_FIELDS:
        per_field[field] = _set_jaccard(features_a.get(field), features_b.get(field))
    for field in _EXACT_FIELDS:
        per_field[field] = _exact_match(features_a.get(field), features_b.get(field))
    for field in _TOKEN_FIELDS:
        per_field[field] = _token_jaccard(features_a.get(field), features_b.get(field))
    overall = sum(per_field.values()) / len(per_field) if per_field else 0.0
    return {'overall': overall, 'per_field': per_field}


def eval_structured_features(image_input, reference_features: dict,
                              slug: str = '_eval_') -> float:
    '''Quality signal: extract 13-field structured features from a candidate
    image (via gpt-4o vision) and compute equal-weighted per-field agreement
    against a reference feature dict.

    Used by the agent loop when --quality-signal=structured_features. The
    reference_features comes from a pre-extracted source like
    `data/{slug}/structured_features_initial_v1.json`.

    The vision-extraction call is cached via the existing
    `structured_features` cache namespace. For the cache to share with the
    post-hoc `extract_structured_features.py --source generated` flow, the
    `slug` parameter MUST match the product slug used by that flow (because
    slug appears both in the cache-key inputs and in the prompt header that
    gpt-4o sees, so different slugs produce different keys and possibly
    different outputs).

    image_input: PIL.Image or path.
    reference_features: dict in the 13-field structured-features schema.
    slug: product slug (e.g., 'water_bottle'). Defaults to '_eval_' for
          callers that don't have a slug handy; pass the real slug if you
          want cache sharing with --source generated.
    Returns: float in [0, 1].'''
    # Lazy import to avoid an eval_image <-> extract_structured_features cycle
    # at module load time. extract_structured_features imports compImage from
    # this module via the test path, but only at runtime.
    from extract_structured_features import extract_features_from_pil

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')

    extracted = extract_features_from_pil(img, slug=slug)
    return feature_agreement(extracted, reference_features)


## VLM-AS-JUDGE QUALITY SIGNAL ##
#
# Used by the agent loop when --quality-signal=vlm_judge. Scores
# how faithfully a generated image depicts a target product versus
# a ground-truth reference photograph, using gpt-4o vision as a
# direct judge. Returns a float in [0, 1].
#
# This signal is LEAKY by design — it compares generated images
# directly against ground-truth photographs. Documented as a
# controlled opt-in violation of the no-leakage boundary in
# docs/handoff/05_decisions_log.md.
#
# The prompt is explicitly tuned to score product-level similarity
# and IGNORE composition / framing / background / styling. This
# reduces the signal's tendency to reward compositional shifts
# toward the ground truth photograph's styling (lifestyle vs.
# retail layout) rather than actual product rendering improvement.

import base64
import hashlib
import json
import mimetypes
import os

VLM_JUDGE_MODEL = 'gpt-4o'
VLM_JUDGE_TEMPERATURE = 0
VLM_JUDGE_MAX_TOKENS = 400

VLM_JUDGE_SYSTEM = (
    'You are a careful product-image evaluator. Your job is to judge how '
    'faithfully a generated image depicts a target product, regardless of '
    'how the product is staged or presented. Ground your score in '
    'product-level visual evidence; use the full 0-100 range as the '
    'evidence warrants. Do not anchor to round numbers.'
)

VLM_JUDGE_PROMPT = '''You will see two images of a product.
Image A: the actual product (reference photograph).
Image B: a generated image attempting to depict the same product.

Your job is to rate how faithfully Image B depicts the product itself,
regardless of how it is staged or presented.

Focus ONLY on the product's intrinsic visual properties:
  - Color, wash, finish, material appearance
  - Shape, silhouette, proportions
  - Texture and construction details (stitching, hardware, patterns)
  - Logos, branding, text on the product itself
  - Distinctive features

IGNORE these (they are NOT product-level differences):
  - Composition and framing (retail layout vs lifestyle vs studio)
  - Background, props, lighting, camera angle
  - Whether a model is present
  - Image cropping, aspect ratio
  - Text overlays, watermarks, marketing elements

Step 1: List 3-5 product-level differences between B's and A's
depiction of the product. If the products are depicted equivalently
despite different framing, the list may be very short.

Step 2: Rate how faithfully Image B depicts the same product as
Image A, as an INTEGER from 0 to 100:
  0   = wrong product entirely
  100 = visually identical product depiction

Your score should reflect how much of the product's identity
(color, shape, material, details, branding) Image B preserves.

Use the full 0-100 range. Different image pairs should produce
distinctly different scores -- do NOT cluster at round numbers
like 50, 65, 70, 75. Scores like 37, 52, 63, 81 are preferred
over 25, 50, 75 when the evidence supports them.

Return the score as an integer.

JSON only:
{
  "differences": ["diff1", "diff2", "..."],
  "score": <integer 0-100>
}
'''


def _image_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = 'image/jpeg'
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'data:{mime};base64,{b64}'


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def vlm_judge_eval(image_input, ground_truth_image_path: str,
                    slug: str = '_eval_') -> float:
    '''Quality signal: ask gpt-4o vision to score how faithfully a
    generated image depicts the same product as a ground-truth
    reference photograph. Returns a float in [0, 1].

    Used by the agent loop when --quality-signal=vlm_judge. The prompt
    is tuned to score product-level similarity and explicitly ignore
    composition / framing / background / styling differences.

    image_input: PIL.Image or path to generated image.
    ground_truth_image_path: path to the ground-truth product photo
                              (typically data/{slug}/images/main.jpg).
    slug: product slug for telemetry; included in the cache-key inputs
          so different products don't cross-pollinate the cache.
    Returns: float in [0, 1].
    '''
    # Lazy import to avoid a top-of-module circular with openai.
    from openai import OpenAI

    if not os.path.exists(ground_truth_image_path):
        raise FileNotFoundError(
            f'ground truth image missing: {ground_truth_image_path}')

    # Resolve the generated image to bytes for hashing + to a path for the
    # data URL. PIL.Image inputs get written to a temp file.
    if isinstance(image_input, (str, Path)):
        generated_path = str(image_input)
        with open(generated_path, 'rb') as f:
            gen_bytes = f.read()
    else:
        import io as _io
        buf = _io.BytesIO()
        image_input.convert('RGB').save(buf, format='PNG')
        gen_bytes = buf.getvalue()
        # Write to a deterministic temp path for the data URL.
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            prefix=f'vlm_judge_{slug}_', suffix='.png', delete=False)
        tmp.write(gen_bytes)
        tmp.close()
        generated_path = tmp.name

    with open(ground_truth_image_path, 'rb') as f:
        gt_bytes = f.read()

    inputs = {
        'kind': 'vlm_judge',
        'slug': slug,
        'model': VLM_JUDGE_MODEL,
        'temperature': VLM_JUDGE_TEMPERATURE,
        'system_prompt': VLM_JUDGE_SYSTEM,
        'user_prompt': VLM_JUDGE_PROMPT,
        'ground_truth_sha256': _sha256_bytes(gt_bytes),
        'generated_sha256': _sha256_bytes(gen_bytes),
    }

    def _live() -> dict:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=VLM_JUDGE_MODEL,
            temperature=VLM_JUDGE_TEMPERATURE,
            max_tokens=VLM_JUDGE_MAX_TOKENS,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': VLM_JUDGE_SYSTEM},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': VLM_JUDGE_PROMPT},
                        {'type': 'image_url',
                         'image_url': {'url': _image_data_url(ground_truth_image_path)}},
                        {'type': 'image_url',
                         'image_url': {'url': _image_data_url(generated_path)}},
                    ],
                },
            ],
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        raw = parsed.get('score', 0)
        try:
            score = float(raw) / 100.0
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        return {
            'differences': parsed.get('differences', []),
            'score': score,
            'score_raw_integer': raw,
        }

    result = cached_call('vlm_judge', inputs, _live, format='json')
    return float(result['score'])
