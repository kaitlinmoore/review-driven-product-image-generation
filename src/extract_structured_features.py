''' extract_structured_features.py
------------------------------
Extracts a fixed-schema JSON summary of a product's visual features from one
of four sources: the initial prompt, the converged prompt after iterative refinement,
the ground-truth product image(s), or the converged generated image. The same schema is used
for all four, so the report can line them up side by side as a comparative evaluation tool.

Inputs (one of):
    --source initial       → data/{product}/initial_prompt.txt
    --source converged     → data/{product}/converged_prompt.txt   
    --source ground_truth  → data/{product}/images/main.jpg         (GPT-4o vision)
    --source generated     → --image-path <file>                    (GPT-4o vision)

Outputs:
    data/{product}/structured_features_{source}_v1.json
    data/{product}/structured_features_{source}_v1_meta.json

Schema (13 fields, two enums + free text):
    product_type, primary_colors[], materials[], finish, texture,
    size_descriptor (enum), measurements, shape_and_form,
    decorative_elements[], visible_parts[],
    brand_visibility (enum), brand_description, overall_aesthetic

Usage and CLI flags:
    python src/extract_structured_features.py --source initial
    python src/extract_structured_features.py --source converged  
    python src/extract_structured_features.py --source ground_truth --only chess # Chess used as test set.
    python src/extract_structured_features.py --source generated --only chess --image-path <path>
'''

import argparse
import base64
import datetime
import hashlib
import json
import mimetypes
import os
import sys
import time

# Use for local .env-defined OpenAI API key.
from dotenv import load_dotenv

from replay import cached_call, path_hash, image_hash
from build_prompt_context import (
    clean_text,
    flatten_description,
    load_top_ranked,
    DEFAULT_TOP_N,
)

# Try to prevent encoding errors.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

## CONSTANTS ##
DATA_DIR = 'data'

LLM_MODEL = 'gpt-4o'
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 800

# GPT-4o Pricing (TODO: Confirm this is current.)
LLM_INPUT_PRICE = 2.50 / 1_000_000
LLM_OUTPUT_PRICE = 10.00 / 1_000_000
COST_ABORT_THRESHOLD = 2.00


# ============================================================================
# v1 STRUCTURED-EXTRACTION SYSTEM PROMPT | DO NOT EDIT WITHOUT BUMPING VERSION.
#
# prompt_version and prompt_sha256 are written into every *_meta.json sidecar.
# If you edit this text, bump to v2 so past outputs stay traceable. This is 
# crtical for functional replay.
# ============================================================================
EXTRACTION_PROMPT_VERSION = 'v1'

EXTRACTION_PROMPT_SYSTEM_V1 = '''You are extracting structured visual features from a product. Return a single JSON object with EXACTLY the fields and types specified. Focus only on the product's visual appearance.

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
- Output must be valid JSON with exactly these 13 keys. No preamble, no comments, no extra keys.'''

EXTRACTION_PROMPT_SHA256 = hashlib.sha256(EXTRACTION_PROMPT_SYSTEM_V1.encode('utf-8')).hexdigest()


REQUIRED_SCHEMA_KEYS = {
    'product_type', 'primary_colors', 'materials', 'finish', 'texture',
    'size_descriptor', 'measurements', 'shape_and_form',
    'decorative_elements', 'visible_parts',
    'brand_visibility', 'brand_description', 'overall_aesthetic',
}
ENUM_VALUES = {
    'size_descriptor': {'tiny', 'small', 'medium', 'large', 'oversized', 'unspecified'},
    'brand_visibility': {'prominent', 'subtle', 'absent'},
}

# JSON schema passed to the OpenAI Structured Outputs endpoint. `strict: True`
# means the model is constrained at decoding time, so enum violations are
# impossible by construction, not just discouraged by the system prompt.
STRUCTURED_FEATURES_SCHEMA_V1 = {
    'type': 'object',
    'additionalProperties': False,
    'required': sorted(REQUIRED_SCHEMA_KEYS),
    'properties': {
        'product_type':         {'type': 'string'},
        'primary_colors':       {'type': 'array', 'items': {'type': 'string'}},
        'materials':            {'type': 'array', 'items': {'type': 'string'}},
        'finish':               {'type': 'string'},
        'texture':              {'type': 'string'},
        'size_descriptor':      {'type': 'string', 'enum': sorted(ENUM_VALUES['size_descriptor'])},
        'measurements':         {'type': ['string', 'null']},
        'shape_and_form':       {'type': 'string'},
        'decorative_elements':  {'type': 'array', 'items': {'type': 'string'}},
        'visible_parts':        {'type': 'array', 'items': {'type': 'string'}},
        'brand_visibility':     {'type': 'string', 'enum': sorted(ENUM_VALUES['brand_visibility'])},
        'brand_description':    {'type': ['string', 'null']},
        'overall_aesthetic':    {'type': 'string'},
    },
}


## SOURCE HANDLING ##

SOURCES = (
    'initial',          # synthesized visual description (generate_initial_prompt output)
    'converged',        # orchestration loop output (pending Phase 4 run)
    'prompt_context',   # raw metadata + top-N reviews (pre-synthesis)
    'metadata_only',    # just the listing fields (no reviews)
    'reviews_only',     # just the top-N reviews (no listing metadata)
    'ground_truth',     # all product images for the listing (via gpt-4o vision, multi-image)
    'generated',        # a pipeline-generated image (via gpt-4o vision, single image)
)

# Per-product image exclusions for the ground_truth extraction. These files
# exist on disk (they're real Amazon listing images we want to keep as
# artifacts of what the listing contains) but are excluded from the VLM call
# because they'd distort the structured features of the product ITSELF.
# Reasons are in the comments. Audit new products by inspecting the output
# JSON after a run; add entries here if pollution shows up.
GROUND_TRUTH_IMAGE_EXCLUDES: dict[str, set[str]] = {
    'water_bottle': {
        'alt_02.jpg',   # infographic with feature callouts (text-heavy)
        'alt_03.jpg',   # dimension guide with full specs table
        'alt_04.jpg',   # lifestyle shot — cream cable-knit sweater dominates
        'alt_05.jpg',   # lifestyle shot — puppy + red harness dominate
        'alt_06.jpg',   # lifestyle shot — woman + large dog + autumn scene dominate
    },
    'headphones': {
        'alt_05.jpg',   # carrying case only, no headphones visible
    },
    # Other products: none excluded yet. Re-audit after the first run if any
    # output shows non-product content leaking into primary_colors / texture / etc.
}

ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def _list_product_images(slug: str, images_dir: str) -> list[str]:
    '''Return sorted full paths to ground-truth images for a product, after
    applying GROUND_TRUTH_IMAGE_EXCLUDES. Deterministic order (sorted by
    filename) so the cache key is stable.'''
    if not os.path.isdir(images_dir):
        return []
    excludes = GROUND_TRUTH_IMAGE_EXCLUDES.get(slug, set())
    out = []
    for name in sorted(os.listdir(images_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            continue
        if name in excludes:
            continue
        out.append(os.path.join(images_dir, name))
    return out

# Reviews count used for reviews_only source. Matches build_prompt_context's
# default so the reviews_only comparison lines up with prompt_context.
REVIEWS_ONLY_TOP_N = DEFAULT_TOP_N


def source_artifact_path(slug: str, source: str, pdir: str,
                          image_path_override: str | None = None,
                          config_name: str | None = None,
                          model_tag: str | None = None) -> str | None:
    '''Return the expected filesystem path for a product/source pair.
    Returns None for 'generated' when no --image-path is supplied and no
    (config_name, model_tag) pair is provided to derive the path.

    When config_name AND model_tag are both provided, the per-config
    artifacts from the agent loop are read instead of canonical pointers:
    - source=converged: `converged_prompt_{model}_{config}.txt`
    - source=generated: `generated_image_{model}_{config}.png`

    For metadata_only and reviews_only, the path points at the source-of-truth
    file on disk (metadata.json or reviews_ranked.jsonl); the actual text
    sent to the LLM is constructed from that file's contents at call time.'''
    per_config = config_name is not None and model_tag is not None
    if source == 'initial':
        return os.path.join(pdir, 'initial_prompt.txt')
    if source == 'converged':
        if per_config:
            return os.path.join(pdir, f'converged_prompt_{model_tag}_{config_name}.txt')
        return os.path.join(pdir, 'converged_prompt.txt')
    if source == 'prompt_context':
        return os.path.join(pdir, 'prompt_context.txt')
    if source == 'metadata_only':
        return os.path.join(pdir, 'metadata.json')
    if source == 'reviews_only':
        return os.path.join(pdir, 'reviews_ranked.jsonl')
    if source == 'ground_truth':
        # Directory of images (multiple per product). _list_product_images
        # enumerates the files and applies GROUND_TRUTH_IMAGE_EXCLUDES.
        return os.path.join(pdir, 'images')
    if source == 'generated':
        if image_path_override is not None:
            return image_path_override
        if per_config:
            return os.path.join(pdir, f'generated_image_{model_tag}_{config_name}.png')
        return None
    raise ValueError(f'unknown source: {source}')


def source_is_image(source: str) -> bool:
    return source in {'ground_truth', 'generated'}


## SOURCE TEXT BUILDERS ##
# For sources that don't have a single pre-built text file, we construct the
# LLM input text here from the underlying data. Kept parallel to the layout
# that build_prompt_context.py produces, so reviews_only is a fair
# subset-comparison to prompt_context (same HTML cleanup, same formatting).

def _build_metadata_only_text(metadata: dict) -> str:
    '''Format metadata.json as LLM input: title/brand/category/reception/price/
    features/description. No reviews.'''
    lines: list[str] = []
    lines.append('=== PRODUCT METADATA ===')
    title = (metadata.get('title') or '').strip()
    if title:
        lines.append(f'Title: {title}')
    store = (metadata.get('store') or '').strip()
    if store:
        lines.append(f'Brand: {store}')
    main_cat = (metadata.get('main_category') or '').strip()
    if main_cat:
        lines.append(f'Category: {main_cat}')
    avg = metadata.get('average_rating')
    n_ratings = metadata.get('rating_number')
    if avg is not None and n_ratings is not None:
        lines.append(f'Reception: {avg}/5 stars across {n_ratings:,} ratings')
    price = metadata.get('price')
    if price is not None:
        lines.append(f'Price: ${price}')

    features = metadata.get('features') or []
    if features:
        lines.append('')
        lines.append('=== FEATURES (from listing) ===')
        for f in features:
            f_str = str(f).strip()
            if f_str:
                lines.append(f'- {f_str}')

    desc = flatten_description(metadata.get('description'))
    if desc:
        lines.append('')
        lines.append('=== DESCRIPTION (from listing) ===')
        lines.append(desc)

    return '\n'.join(lines) + '\n'


def _build_reviews_only_text(reviews: list[dict]) -> str:
    '''Format the top-N ranked reviews as LLM input. No metadata block.'''
    lines: list[str] = []
    lines.append(f'=== TOP {len(reviews)} REVIEWS (ranked by helpful_vote DESC) ===')
    for r in reviews:
        rating = int(r.get('rating', 0) or 0)
        hv = r.get('helpful_vote', 0) or 0
        rank = r.get('_rank', '?')
        title = clean_text(r.get('title') or '')
        body = clean_text(r.get('text') or '')
        lines.append('')
        lines.append(f'[Rank {rank} | {rating}\u2605 | {hv} helpful votes]')
        if title:
            lines.append(f'Title: {title}')
        if body:
            lines.append(f'Body: {body}')
    return '\n'.join(lines) + '\n'


def output_paths(slug: str, pdir: str, source: str,
                  config_name: str | None = None,
                  model_tag: str | None = None) -> tuple[str, str]:
    '''Compute the JSON + meta output paths.

    When config_name AND model_tag are provided AND source is
    converged/generated, the output filenames carry a {model}_{config}
    suffix so per-config Phase 4 extractions don't overwrite each other.
    Otherwise the canonical naming is used.'''
    per_config = (config_name is not None and model_tag is not None
                   and source in ('converged', 'generated'))
    if per_config:
        base = os.path.join(
            pdir,
            f'structured_features_{source}_{model_tag}_{config_name}_{EXTRACTION_PROMPT_VERSION}'
        )
    else:
        base = os.path.join(pdir, f'structured_features_{source}_{EXTRACTION_PROMPT_VERSION}')
    return base + '.json', base + '_meta.json'


## DISCOVERY ##

def discover_products() -> dict[str, str]:
    '''Return {slug: product_dir} for every data/*/ dir with metadata.json.
    Source-specific availability is handled per-product at run time so we can
    report 'skipped: source artifact missing' for each source cleanly.'''
    out: dict[str, str] = {}
    if not os.path.isdir(DATA_DIR):
        return out
    for name in sorted(os.listdir(DATA_DIR)):
        pdir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(pdir) or name == 'filter_caches':
            continue
        if os.path.exists(os.path.join(pdir, 'metadata.json')):
            out[name] = pdir
    return out


def resolve_only(products, needle: str) -> dict[str, str]:
    matches = [k for k in products if needle.lower() in k.lower()]
    if len(matches) == 0:
        print(f'[!] --only {needle!r} matches no product. Available: {sorted(products)}')
        sys.exit(1)
    if len(matches) > 1:
        print(f'[!] --only {needle!r} matches multiple: {matches}.')
        sys.exit(1)
    return {matches[0]: products[matches[0]]}


## LLM CALLS (Text and Vision) ##

def image_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = 'image/jpeg'
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'data:{mime};base64,{b64}'


def _pil_to_data_url(pil_image, fmt: str = 'PNG') -> str:
    '''Encode a PIL.Image directly to a data URL (no disk roundtrip).
    Used by extract_features_from_pil for in-loop callers that hold a PIL
    image in memory.'''
    import io
    buf = io.BytesIO()
    pil_image.convert('RGB').save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    mime = 'image/png' if fmt.upper() == 'PNG' else 'image/jpeg'
    return f'data:{mime};base64,{b64}'


def _build_text_for_source(source: str, artifact_path: str) -> str:
    '''Assemble the LLM-input text for a non-image source.

    - initial / converged / prompt_context: read the file directly.
    - metadata_only: load metadata.json, format without reviews.
    - reviews_only: load top-N rows of reviews_ranked.jsonl, format without
      the metadata block.'''
    if source == 'metadata_only':
        with open(artifact_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return _build_metadata_only_text(metadata)
    if source == 'reviews_only':
        reviews = load_top_ranked(artifact_path, REVIEWS_ONLY_TOP_N)
        return _build_reviews_only_text(reviews)
    # initial / converged / prompt_context: direct file read.
    with open(artifact_path, 'r', encoding='utf-8') as f:
        return f.read()


def call_llm(client, source: str, artifact_path: str, slug: str) -> tuple[dict, dict]:
    '''Returns (parsed_dict, usage_dict).

    Replay-aware: text sources key on the source-specific text content
    (either file contents or constructed text); image sources key on
    path_hash() of the image bytes so the cache survives across machines
    with different absolute paths but identical artifact content.'''
    # Build the inputs dict that uniquely identifies this call for caching.
    # Image sources hash the bytes; text sources include the constructed text.
    image_paths: list[str] = []
    if source == 'ground_truth':
        image_paths = _list_product_images(slug, artifact_path)
        if not image_paths:
            raise RuntimeError(
                f'No included ground-truth images for {slug} under {artifact_path} '
                f'(after applying GROUND_TRUTH_IMAGE_EXCLUDES).'
            )
        # Per-file hash + basenames so the cache key invalidates cleanly when
        # any included image changes, is added, or is removed.
        source_input = {
            'image_hashes': [path_hash(p) for p in image_paths],
            'image_basenames': [os.path.basename(p) for p in image_paths],
        }
        text = None
    elif source == 'generated':
        image_paths = [artifact_path]
        # Pixel-content hash (not file-byte hash) so the cache shares with
        # the in-loop `eval_structured_features` call site, which receives a
        # PIL.Image directly and has no file path to work with at the moment
        # of extraction.
        source_input = {'image_hash': image_hash(artifact_path)}
        text = None
    else:
        text = _build_text_for_source(source, artifact_path)
        source_input = {'text': text}

    inputs = {
        'model': LLM_MODEL,
        'temperature': LLM_TEMPERATURE,
        'max_tokens': LLM_MAX_TOKENS,
        'prompt_version': EXTRACTION_PROMPT_VERSION,
        'prompt_sha256': EXTRACTION_PROMPT_SHA256,
        'schema_name': 'structured_features_v1',
        'source': source,
        'slug': slug,
        **source_input,
    }

    def _live() -> dict:
        if source_is_image(source):
            # Single intro text, followed by one image_url part per included image.
            # For ground_truth the VLM is asked to aggregate across views of the
            # same product; for generated we're showing one image.
            if source == 'ground_truth' and len(image_paths) > 1:
                header = (
                    f'Product slug: {slug}. You are being shown multiple views '
                    f'of the SAME product from its listing. Extract structured '
                    f'visual features that characterize the product itself, '
                    f'aggregating across the views. Return JSON per the schema.'
                )
            else:
                header = (
                    f'Product slug: {slug}. Extract the structured visual features '
                    f'from this product image and return them as JSON per the schema.'
                )
            user_content = [{'type': 'text', 'text': header}]
            for p in image_paths:
                user_content.append({
                    'type': 'image_url',
                    'image_url': {'url': image_data_url(p)},
                })
        else:
            user_content = (
                f'Product slug: {slug}. Extract the structured visual features from the following '
                f'product description and return them as JSON per the schema.\n\n{text}'
            )

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'structured_features_v1',
                    'strict': True,
                    'schema': STRUCTURED_FEATURES_SCHEMA_V1,
                },
            },
            messages=[
                {'role': 'system', 'content': EXTRACTION_PROMPT_SYSTEM_V1},
                {'role': 'user', 'content': user_content},
            ],
        )
        raw = (resp.choices[0].message.content or '').strip()
        parsed = json.loads(raw)  # strict schema guarantees shape + enum compliance
        usage = {
            'input_tokens': resp.usage.prompt_tokens if resp.usage else 0,
            'output_tokens': resp.usage.completion_tokens if resp.usage else 0,
        }
        return {'parsed': parsed, 'usage': usage}

    result = cached_call('structured_features', inputs, _live, format='json')
    return result['parsed'], result['usage']


def extract_features_from_pil(pil_image, slug: str = '_eval_') -> dict:
    '''Extract the 13-field structured-features dict from a PIL.Image directly.

    Used by the in-loop quality signal (`eval_image.eval_structured_features`)
    so we don't have to round-trip the candidate image through disk first.
    Cache namespace and inputs dict are constructed identically to the
    `--source generated` path in call_llm() — both call sites compute the
    same cache key for the same image, so a generated image scored in-loop
    AND analyzed post-hoc via `--source generated` shares one cache entry.

    Lazy OpenAI client construction so test/import paths that never hit live
    don't require OPENAI_API_KEY in the environment.

    pil_image: PIL.Image (any mode; converted to RGB internally).
    slug: short tag stamped into the cache-key inputs and into the user
          prompt (purely informational; does not change extraction behavior).
          Defaults to '_eval_' for in-loop callers that don't have a slug
          handy.
    Returns: parsed feature dict.'''
    inputs = {
        'model': LLM_MODEL,
        'temperature': LLM_TEMPERATURE,
        'max_tokens': LLM_MAX_TOKENS,
        'prompt_version': EXTRACTION_PROMPT_VERSION,
        'prompt_sha256': EXTRACTION_PROMPT_SHA256,
        'schema_name': 'structured_features_v1',
        'source': 'generated',
        'slug': slug,
        'image_hash': image_hash(pil_image),
    }

    def _live() -> dict:
        load_dotenv()
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY not set; cannot run live extraction.')
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError('`openai` package not installed.')
        client = OpenAI(api_key=api_key)

        header = (
            f'Product slug: {slug}. Extract the structured visual features '
            f'from this product image and return them as JSON per the schema.'
        )
        user_content = [
            {'type': 'text', 'text': header},
            {'type': 'image_url', 'image_url': {'url': _pil_to_data_url(pil_image)}},
        ]
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'structured_features_v1',
                    'strict': True,
                    'schema': STRUCTURED_FEATURES_SCHEMA_V1,
                },
            },
            messages=[
                {'role': 'system', 'content': EXTRACTION_PROMPT_SYSTEM_V1},
                {'role': 'user', 'content': user_content},
            ],
        )
        raw = (resp.choices[0].message.content or '').strip()
        parsed = json.loads(raw)
        usage = {
            'input_tokens': resp.usage.prompt_tokens if resp.usage else 0,
            'output_tokens': resp.usage.completion_tokens if resp.usage else 0,
        }
        return {'parsed': parsed, 'usage': usage}

    result = cached_call('structured_features', inputs, _live, format='json')
    return result['parsed']


## VALIDATION ##

def validate_schema(parsed: dict) -> list[str]:
    '''Return a list of warnings (empty if clean). Doesn't raise.'''
    warnings: list[str] = []
    missing = REQUIRED_SCHEMA_KEYS - parsed.keys()
    extra = parsed.keys() - REQUIRED_SCHEMA_KEYS
    if missing:
        warnings.append(f'missing keys: {sorted(missing)}')
    if extra:
        warnings.append(f'unexpected keys: {sorted(extra)}')
    for field, allowed in ENUM_VALUES.items():
        v = parsed.get(field)
        if v is not None and v not in allowed:
            warnings.append(f'{field}={v!r} not in enum {sorted(allowed)}')
    # Light type checks for the list fields
    for field in ('primary_colors', 'materials', 'decorative_elements', 'visible_parts'):
        v = parsed.get(field)
        if v is not None and not isinstance(v, list):
            warnings.append(f'{field} should be a list, got {type(v).__name__}')
    return warnings


## COST ESTIMATE ##

def estimate_cost(n_calls: int, source: str, avg_input_chars: int = 6000) -> tuple[float, float, float]:
    sys_tokens = len(EXTRACTION_PROMPT_SYSTEM_V1) / 4
    # Text sources: ~1.5K tokens of user text (initial_prompt.txt is ~250 words).
    # Image sources: add ~1.1K tokens per image tile (gpt-4o default detail).
    # ground_truth is multi-image (typically 4-8 per product after exclusions);
    # estimate at 5 tiles on average for rough budget reporting.
    per_call_in = sys_tokens + avg_input_chars / 4
    if source == 'ground_truth':
        per_call_in += 1100 * 5
    elif source == 'generated':
        per_call_in += 1100
    est_in = n_calls * per_call_in
    est_out = n_calls * 500
    est_usd = est_in * LLM_INPUT_PRICE + est_out * LLM_OUTPUT_PRICE
    return est_in, est_out, est_usd


## PER-PRODUCT DRIVER ##

def process(slug: str, pdir: str, source: str, client, args) -> dict:
    artifact_path = source_artifact_path(
        slug, source, pdir,
        image_path_override=args.image_path,
        config_name=getattr(args, 'config_name', None),
        model_tag=getattr(args, 'model', None),
    )
    if artifact_path is None:
        return {'slug': slug, 'skipped': True, 'reason': 'no --image-path or --config-name+--model for source=generated'}
    if not os.path.exists(artifact_path):
        return {'slug': slug, 'skipped': True, 'reason': f'missing input: {artifact_path}'}

    out_json, out_meta = output_paths(
        slug, pdir, source,
        config_name=getattr(args, 'config_name', None),
        model_tag=getattr(args, 'model', None),
    )
    if os.path.exists(out_json) and not args.force:
        return {'slug': slug, 'skipped': True, 'reason': 'output exists (use --force)'}

    t0 = time.time()
    try:
        parsed, usage = call_llm(client, source, artifact_path, slug)
    except Exception as e:
        print(f'  [{slug}] [!] extraction failed: {type(e).__name__}: {e}')
        return {'slug': slug, 'error': str(e)}

    elapsed = time.time() - t0
    cost = usage['input_tokens'] * LLM_INPUT_PRICE + usage['output_tokens'] * LLM_OUTPUT_PRICE
    warnings = validate_schema(parsed)

    # Atomic write of the features JSON itself.
    tmp = out_json + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_json)

    # Provenance sidecar
    meta = {
        'slug': slug,
        'source': source,
        'source_artifact': artifact_path,
        'model': LLM_MODEL,
        'temperature': LLM_TEMPERATURE,
        'max_tokens': LLM_MAX_TOKENS,
        'prompt_version': EXTRACTION_PROMPT_VERSION,
        'prompt_sha256': EXTRACTION_PROMPT_SHA256,
        'input_tokens': usage['input_tokens'],
        'output_tokens': usage['output_tokens'],
        'cost_usd': round(cost, 5),
        'elapsed_seconds': round(elapsed, 2),
        'schema_warnings': warnings,
        'timestamp_utc': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds').replace('+00:00', 'Z'),
    }
    with open(out_meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    warn_str = f'  ⚠ {len(warnings)} warnings' if warnings else ''
    print(f'  [{slug}/{source}] wrote {out_json} '
          f'({usage["input_tokens"]}+{usage["output_tokens"]} tok, ${cost:.4f}, {elapsed:.1f}s){warn_str}')
    return {'slug': slug, 'source': source, 'cost_usd': cost, 'warnings': warnings, **meta}


## MAIN ##

def main():
    parser = argparse.ArgumentParser(description='Extract structured visual features per product.')
    parser.add_argument('--source', choices=SOURCES, required=True,
                        help='Which artifact to extract from.')
    parser.add_argument('--only', default=None,
                        help='Substring-match a single product slug.')
    parser.add_argument('--image-path', default=None,
                        help='For --source generated: path to the generated image file. '
                             'Optional if --config-name and --model are set (auto-derives '
                             'data/{slug}/generated_image_{model}_{config}.png).')
    parser.add_argument('--config-name', default=None,
                        help='For --source converged or generated: read the per-config '
                             'agent-loop artifact (converged_prompt_{model}_{config}.txt or '
                             'generated_image_{model}_{config}.png) and write to a '
                             'correspondingly-suffixed output. Requires --model.')
    parser.add_argument('--model', choices=['flux', 'gpt'], default=None,
                        help='Required when --config-name is set. Which image-model '
                             'flavor of the per-config artifact to extract from.')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if output exists.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show prompt + cost estimate, no API calls.')
    args = parser.parse_args()

    # Validate the (--config-name, --model) pairing.
    if (args.config_name is None) != (args.model is None):
        print('[!] --config-name and --model must be provided together.')
        sys.exit(1)
    if args.config_name is not None and args.source not in ('converged', 'generated'):
        print(f'[!] --config-name only applies to --source converged or generated; '
              f'got --source={args.source!r}.')
        sys.exit(1)

    if args.source == 'generated' and not args.only:
        print('[!] --source generated requires --only <slug> (and --image-path or --config-name+--model).')
        sys.exit(1)
    if args.source == 'generated' and not args.image_path and not args.config_name:
        print('[!] --source generated requires --image-path OR (--config-name AND --model).')
        sys.exit(1)

    products = discover_products()
    if not products:
        print(f'[!] No products discovered under {DATA_DIR}/.')
        sys.exit(1)

    selected = resolve_only(products, args.only) if args.only else products
    print(f'Source: {args.source}')
    print(f'Discovered products: {sorted(products)}')
    if args.only:
        print(f'Selected via --only: {list(selected)}')

    # Pre-flight: count how many will actually run.
    to_do: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    for slug, pdir in selected.items():
        artifact = source_artifact_path(
            slug, args.source, pdir,
            image_path_override=args.image_path,
            config_name=args.config_name,
            model_tag=args.model,
        )
        out_json, _ = output_paths(
            slug, pdir, args.source,
            config_name=args.config_name,
            model_tag=args.model,
        )
        if artifact is None:
            skipped.append((slug, 'no --image-path'))
            continue
        if not os.path.exists(artifact):
            skipped.append((slug, f'missing: {artifact}'))
            continue
        if os.path.exists(out_json) and not args.force:
            skipped.append((slug, 'output exists (use --force)'))
            continue
        to_do.append((slug, pdir))

    if skipped:
        print(f'\nSkipping {len(skipped)}:')
        for slug, reason in skipped:
            print(f'  [{slug}] {reason}')

    if not to_do:
        print('\nNothing to do.')
        return

    est_in, est_out, est_usd = estimate_cost(len(to_do), args.source)
    print(f'\nTo process: {[s for s, _ in to_do]}')
    print(f'Estimated cost: ${est_usd:.4f}  (in~{est_in:.0f} tok, out~{est_out:.0f} tok)')

    if args.dry_run:
        print('\n' + '=' * 72)
        print(f'  v{EXTRACTION_PROMPT_VERSION} SYSTEM PROMPT (sha256={EXTRACTION_PROMPT_SHA256[:16]}...)')
        print('=' * 72)
        print(EXTRACTION_PROMPT_SYSTEM_V1)
        print('\nDry-run complete. Re-run without --dry-run to call the LLM.')
        return

    if est_usd > COST_ABORT_THRESHOLD:
        print(f'[!] Estimated ${est_usd:.2f} exceeds ${COST_ABORT_THRESHOLD} guard. Aborting.')
        sys.exit(1)

    load_dotenv()
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print('[!] OPENAI_API_KEY not set.')
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print('[!] `openai` package not installed.')
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    results = []
    total_cost = 0.0
    total_warnings = 0
    for slug, pdir in to_do:
        r = process(slug, pdir, args.source, client, args)
        results.append(r)
        total_cost += r.get('cost_usd', 0.0) if not r.get('skipped') and not r.get('error') else 0.0
        total_warnings += len(r.get('warnings', []) or [])

    print('\n' + '=' * 72)
    print('  DONE')
    print('=' * 72)
    for r in results:
        if r.get('skipped'):
            print(f'  {r["slug"]:<18} SKIPPED ({r["reason"]})')
        elif r.get('error'):
            print(f'  {r["slug"]:<18} ERROR ({r["error"]})')
        else:
            w = len(r.get('warnings', []) or [])
            w_str = f'  ⚠ {w} warnings' if w else ''
            print(f'  {r["slug"]:<18} ${r["cost_usd"]:.4f}  {r["elapsed_seconds"]}s{w_str}')
    print(f'  total cost: ${total_cost:.4f}')
    if total_warnings:
        print(f'  total schema warnings: {total_warnings} (see *_meta.json files)')


if __name__ == '__main__':
    main()
