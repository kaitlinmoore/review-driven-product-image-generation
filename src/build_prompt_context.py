'''build_prompt_context.py
-----------------------
Assembles the INITIAL prompt-generation context per product:
    metadata (title/brand/features/description) and top-N ranked reviews
    → data/{product}/prompt_context.txt

This is used by the one-shot initial-prompt LLM call. It is NOT the dataloader
that PromptWriter iterates over. (That's the full reviews_ranked.jsonl file.)
The initial prompt is the starting point for improvementLoop, which then pulls
more reviews via stepPrompt() to refine prompt descriptiveness.

Why --top 30 default:
- The initial prompt only needs enough reviews to anchor metadata claims and
  add aesthetic/subjective detail. Loop will consume more.
- Keeps the initial-prompt LLM call cheap and fast.
- Overridable via --top N if a product feels starved.

Usage and CLI Flags:
    python build_prompt_context.py                 # all discovered products
    python build_prompt_context.py --only <slug>   # one product
    python build_prompt_context.py --top 50        # override default
'''

import argparse
import html
import json
import os
import re
import sys


try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


DATA_DIR = 'data'
# Must match DEFAULT_SKIP_TOP in reviews_dataloader.py so the initial prompt's
# reviews and the dataloader's reviews don't overlap or leave a gap. Update
# both together if you change this.
DEFAULT_TOP_N = 30


## DISCOVERY ##

def discover_products() -> dict[str, str]:
    '''Products that have metadata.json AND reviews_ranked.jsonl.'''
    out: dict[str, str] = {}
    if not os.path.isdir(DATA_DIR):
        return out
    for name in sorted(os.listdir(DATA_DIR)):
        pdir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(pdir) or name == 'filter_caches':
            continue
        if not os.path.exists(os.path.join(pdir, 'metadata.json')):
            continue
        if not os.path.exists(os.path.join(pdir, 'reviews_ranked.jsonl')):
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


## LOADING ##

def load_metadata(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_top_ranked(path: str, n: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= n:
                break
    return rows


## TEXT ASSEMBLY ##

# HTML tag/break normalization. Applied to anything a human wrote that
# might contain Amazon review HTML (&#34; escapes, <br /> line breaks).
# Intentionally does NOT touch [[ASIN:...]] or [[VIDEOID:...]] markers.
# Those are separate decisions.
_BR_TAG_RE = re.compile(r'<\s*br\s*/?\s*>', re.IGNORECASE)
_HTML_TAG_RE = re.compile(r'<[^>]+>')
_MULTI_BLANK_LINE_RE = re.compile(r'\n{3,}')
_TRAILING_WS_RE = re.compile(r'[ \t]+\n')


def clean_text(s: str) -> str:
    if not s:
        return ''
    s = html.unescape(s)          # &#34; -> ", &amp; -> &, etc.
    s = _BR_TAG_RE.sub('\n', s)   # <br /> or <br> -> newline
    s = _HTML_TAG_RE.sub('', s)   # strip other tags
    s = _TRAILING_WS_RE.sub('\n', s)
    s = _MULTI_BLANK_LINE_RE.sub('\n\n', s)
    return s.strip()


def flatten_description(desc) -> str:
    '''Amazon's description field is a list of strings (multi-paragraph) OR a
    plain string. Normalize to a single string.'''
    if desc is None:
        return ''
    if isinstance(desc, list):
        return clean_text('\n\n'.join(str(s).strip() for s in desc if s))
    return clean_text(str(desc))


def build_context(metadata: dict, reviews: list[dict]) -> str:
    lines: list[str] = []

    # Metadata
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

    # Features
    features = metadata.get('features') or []
    if features:
        lines.append('')
        lines.append('=== FEATURES (from listing) ===')
        for f in features:
            f = str(f).strip()
            if f:
                lines.append(f'- {f}')

    # Description
    desc = flatten_description(metadata.get('description'))
    if desc:
        lines.append('')
        lines.append('=== DESCRIPTION (from listing) ===')
        lines.append(desc)

    # Top Reviews
    lines.append('')
    lines.append(f'=== TOP {len(reviews)} REVIEWS (ranked by helpful_vote DESC) ===')
    for r in reviews:
        rating = int(r.get('rating', 0) or 0)
        hv = r.get('helpful_vote', 0) or 0
        rank = r.get('_rank', '?')
        title = clean_text(r.get('title') or '')
        body = clean_text(r.get('text') or '')
        lines.append('')
        lines.append(f'[Rank {rank} | {rating}★ | {hv} helpful votes]')
        if title:
            lines.append(f'Title: {title}')
        if body:
            lines.append(f'Body: {body}')

    return '\n'.join(lines) + '\n'


## MAIN ##

def process(slug: str, pdir: str, top_n: int) -> dict:
    md = load_metadata(os.path.join(pdir, 'metadata.json'))
    ranked_path = os.path.join(pdir, 'reviews_ranked.jsonl')
    reviews = load_top_ranked(ranked_path, top_n)

    # Count total available for reporting.
    total_available = 0
    with open(ranked_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                total_available += 1

    text = build_context(md, reviews)
    out_path = os.path.join(pdir, 'prompt_context.txt')
    tmp = out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, out_path)

    chars = len(text)
    rough_tokens = chars // 4  # crude estimate
    print(f'[{slug}]')
    print(f'  title          : {(md.get("title") or "")[:80]}')
    print(f'  reviews used   : {len(reviews):>3} / {total_available:,} available '
          f'(capped at --top {top_n})')
    print(f'  context chars  : {chars:>7,}  ~{rough_tokens:,} tokens')
    print(f'  wrote          : {out_path}')
    return {'slug': slug, 'chars': chars, 'tokens_est': rough_tokens,
            'reviews_used': len(reviews), 'reviews_available': total_available}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', default=None,
                        help='Substring-match a single product slug.')
    parser.add_argument('--top', type=int, default=DEFAULT_TOP_N,
                        help=f'Number of top-ranked reviews to include (default {DEFAULT_TOP_N}).')
    args = parser.parse_args()

    products = discover_products()
    if not products:
        print(f'[!] No products found under {DATA_DIR}/ with metadata.json + reviews_ranked.jsonl.')
        sys.exit(1)
    print(f'Discovered products (with ranked reviews): {sorted(products)}')

    selected = resolve_only(products, args.only) if args.only else products

    results = []
    for slug, pdir in selected.items():
        results.append(process(slug, pdir, args.top))

    print()
    print('=' * 72)
    print('  SUMMARY')
    print('=' * 72)
    for r in results:
        print(f'  {r["slug"]:<16} reviews={r["reviews_used"]:>3}/{r["reviews_available"]:<5,}  '
              f'~{r["tokens_est"]:>5,} tokens  ({r["chars"]:,} chars)')


if __name__ == '__main__':
    main()
