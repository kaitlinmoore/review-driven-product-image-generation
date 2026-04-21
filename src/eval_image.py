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
