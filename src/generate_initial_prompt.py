'''generate_initial_prompt.py
--------------------------
One-shot LLM call per product: consumes prompt_context.txt, produces the
free-form visual description that becomes `PromptWriter.self.prompt`, the
"first guess" that the orchestration loop iterates on.

Inputs:
    data/{product}/prompt_context.txt

Outputs:
    data/{product}/initial_prompt.txt          # the description (what feeds diffusion)
    data/{product}/initial_prompt_meta.json    # provenance: model, temp, prompt
                                                  version and SHA, tokens, cost, timestamp

By default, skips products that already have initial_prompt.txt. Use --force
to regenerate.

Usage and CLI Flags:
    python generate_initial_prompt.py --dry-run      # show prompt and cost estimate, no API calls
    python generate_initial_prompt.py                # all products without initial_prompt.txt
    python generate_initial_prompt.py --only <slug>  # one product
    python generate_initial_prompt.py --force        # regenerate everything
'''

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

# Use for local .env defined API keys.
from dotenv import load_dotenv

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

# TODO: Confirm this is current.
LLM_INPUT_PRICE = 2.50 / 1_000_000
LLM_OUTPUT_PRICE = 10.00 / 1_000_000
COST_ABORT_THRESHOLD = 2.00


# ============================================================================
# v1 INITIAL-PROMPT SYSTEM PROMPT | DO NOT EDIT WITHOUT BUMPING VERSION.
#
# The prompt_version string and its sha256 are recorded in the sidecar
# metadata for every generated initial_prompt.txt. If you iterate on this
# prompt text, bump to v2 so past outputs remain traceable to the exact
# system prompt that produced them. This is critical for replay.
# ============================================================================
INITIAL_PROMPT_VERSION = 'v1'

INITIAL_PROMPT_SYSTEM_V1 = '''You are writing a visual description of a product for a text-to-image diffusion model.

Your task: produce a rich, concrete visual description of the product below. Use the metadata (title, features, description) as the factual foundation for claims about dimensions, materials, colors, and construction. Use the reviews to add aesthetic detail, real-world visual impressions, and visual nuances not captured in the listing.

Requirements:
- Focus on what the product LOOKS LIKE. Colors, materials, textures, shapes, proportions, decorative elements, visible construction details, surface finishes.
- Be specific. "Wooden" is weaker than "beech and birch." "Colorful" is weaker than "ochre and deep blue." Use exact terms from the metadata where available.
- Write one coherent description, roughly 150 to 300 words. No bullet lists, no headers.
- Do NOT describe user reactions, price, shipping, or functional behavior (how it performs, how it feels to use) except where those carry visual information (e.g., "pieces skid easily" implies smooth unweighted bottoms).
- Do NOT reference other products mentioned in reviews. If a review compares this product to another, extract only the visual observations about THIS product.
- Do NOT mention star ratings, helpfulness votes, or reviewer identities.

Output: just the visual description. No preamble, no headers, no quotation marks around the whole thing.'''

INITIAL_PROMPT_SHA256 = hashlib.sha256(INITIAL_PROMPT_SYSTEM_V1.encode('utf-8')).hexdigest()


## DISCOVERY ##

def discover_products() -> dict[str, str]:
    '''Products with prompt_context.txt ready.'''
    out: dict[str, str] = {}
    if not os.path.isdir(DATA_DIR):
        return out
    for name in sorted(os.listdir(DATA_DIR)):
        pdir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(pdir) or name == 'filter_caches':
            continue
        if not os.path.exists(os.path.join(pdir, 'prompt_context.txt')):
            continue
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


## LLM CALL ##

def call_llm(client, context_text: str) -> tuple[str, dict]:
    '''Returns (description, usage_dict). Raises on error.'''
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {'role': 'system', 'content': INITIAL_PROMPT_SYSTEM_V1},
            {'role': 'user', 'content': context_text},
        ],
    )
    text = (resp.choices[0].message.content or '').strip()
    usage = {
        'input_tokens': resp.usage.prompt_tokens if resp.usage else 0,
        'output_tokens': resp.usage.completion_tokens if resp.usage else 0,
    }
    return text, usage


## COST ESTIMATE ##

def estimate_cost(context_texts: list[str]) -> tuple[float, float, float]:
    sys_tokens = len(INITIAL_PROMPT_SYSTEM_V1) / 4
    est_in = sum(sys_tokens + len(t) / 4 for t in context_texts)
    est_out = len(context_texts) * 500  # target ~250 words ≈ 375 tokens; budget headroom
    est_usd = est_in * LLM_INPUT_PRICE + est_out * LLM_OUTPUT_PRICE
    return est_in, est_out, est_usd


## PER-PRODUCT DRIVER ##

def process(slug: str, pdir: str, client, force: bool) -> dict:
    ctx_path = os.path.join(pdir, 'prompt_context.txt')
    out_path = os.path.join(pdir, 'initial_prompt.txt')
    meta_path = os.path.join(pdir, 'initial_prompt_meta.json')

    if os.path.exists(out_path) and not force:
        print(f'  [{slug}] initial_prompt.txt already exists — skipping (use --force to regenerate).')
        return {'slug': slug, 'skipped': True}

    with open(ctx_path, 'r', encoding='utf-8') as f:
        context_text = f.read()

    t0 = time.time()
    try:
        description, usage = call_llm(client, context_text)
    except Exception as e:
        print(f'  [{slug}] [!] LLM call failed: {type(e).__name__}: {e}')
        return {'slug': slug, 'error': str(e)}

    elapsed = time.time() - t0
    cost = (usage['input_tokens'] * LLM_INPUT_PRICE
            + usage['output_tokens'] * LLM_OUTPUT_PRICE)

    # Atomic write of the description itself.
    tmp = out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(description + '\n')
    os.replace(tmp, out_path)

    # Provenance sidecar.
    meta = {
        'slug': slug,
        'model': LLM_MODEL,
        'temperature': LLM_TEMPERATURE,
        'max_tokens': LLM_MAX_TOKENS,
        'prompt_version': INITIAL_PROMPT_VERSION,
        'prompt_sha256': INITIAL_PROMPT_SHA256,
        'input_tokens': usage['input_tokens'],
        'output_tokens': usage['output_tokens'],
        'cost_usd': round(cost, 5),
        'elapsed_seconds': round(elapsed, 2),
        'description_chars': len(description),
        'description_words': len(description.split()),
        'timestamp_utc': datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds').replace('+00:00', 'Z'),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f'  [{slug}] wrote {out_path} '
          f'({meta["description_words"]} words, {usage["input_tokens"]}+{usage["output_tokens"]} tok, '
          f'${cost:.4f}, {elapsed:.1f}s)')
    return {'slug': slug, **meta}


## MAIN ##

def main():
    parser = argparse.ArgumentParser(description='Generate initial visual descriptions per product.')
    parser.add_argument('--only', default=None,
                        help='Substring-match a single product slug.')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if initial_prompt.txt already exists.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show system prompt + cost estimate. No API calls.')
    args = parser.parse_args()

    products = discover_products()
    if not products:
        print(f'[!] No products found under {DATA_DIR}/ with prompt_context.txt.')
        sys.exit(1)
    print(f'Discovered products (with context): {sorted(products)}')

    selected = resolve_only(products, args.only) if args.only else products

    # Filter to those that need processing (unless --force).
    to_do: dict[str, str] = {}
    already: list[str] = []
    for slug, pdir in selected.items():
        out_path = os.path.join(pdir, 'initial_prompt.txt')
        if os.path.exists(out_path) and not args.force:
            already.append(slug)
        else:
            to_do[slug] = pdir

    if already:
        print(f'Already have initial_prompt.txt (skipping): {already}')

    if not to_do:
        print('Nothing to do. Use --force to regenerate.')
        return

    # Cost estimate over to-do set
    contexts = []
    for slug, pdir in to_do.items():
        with open(os.path.join(pdir, 'prompt_context.txt'), 'r', encoding='utf-8') as f:
            contexts.append(f.read())
    est_in, est_out, est_usd = estimate_cost(contexts)
    print(f'\nTo process: {list(to_do)}')
    print(f'Estimated cost: ${est_usd:.4f}  (in~{est_in:.0f} tok, out~{est_out:.0f} tok)')

    if args.dry_run:
        print('\n' + '=' * 72)
        print(f'  {INITIAL_PROMPT_VERSION} SYSTEM PROMPT (sha256={INITIAL_PROMPT_SHA256[:16]}...)')
        print('=' * 72)
        print(INITIAL_PROMPT_SYSTEM_V1)
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
    for slug, pdir in to_do.items():
        r = process(slug, pdir, client, force=args.force)
        results.append(r)
        total_cost += r.get('cost_usd', 0.0) if not r.get('skipped') else 0.0

    print('\n' + '=' * 72)
    print('  DONE')
    print('=' * 72)
    for r in results:
        if r.get('skipped'):
            continue
        if r.get('error'):
            print(f'  {r["slug"]:<16} ERROR: {r["error"]}')
            continue
        print(f'  {r["slug"]:<16} {r["description_words"]} words  '
              f'${r["cost_usd"]:.4f}  {r["elapsed_seconds"]}s')
    print(f'  total cost: ${total_cost:.4f}')


if __name__ == '__main__':
    main()
