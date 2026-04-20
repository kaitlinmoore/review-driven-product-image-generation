'''preprocess_reviews.py
---------------------
Consumes raw reviews and the v1 filter-decision cache and produces a clean,
ranked, deduplicated review list per product for Phase 2 to batch-feed from.

Pipeline (Order matters for the reporting numbers. The final set is unchanged.):
    raw  -> dedupe  -> hygiene  -> gate  -> final

  - dedupe: Some raw rows share a user_id and timestamp with another row
    (exact-text duplicates introduced during the Amazon scrape, verified).
    The filter cache collapses these to one decision each; we dedupe here too
    so the ranked output doesn't carry redundant rows.
  - hygiene: Drop text < 50 chars (also catches empty) or > 3000 chars (rants). No language filter.
  - gate: Drop reviews whose cached decision is passes_gate=False.
           Drop reviews whose cached decision is passes_gate=None (LLM errors).
           Drop reviews with NO cached decision (shouldn't happen if cache is complete).
  - ranking: Sort descending by helpful_vote, then by timestamp as tiebreaker.
  - NO top-N truncation. NO stratification.

Inputs:
    data/{product}/reviews.jsonl
    data/filter_caches/{product}_filter_decisions_v1.json

Output:
    data/{product}/reviews_ranked.jsonl  | one JSON object per line:
      { ...original fields..., _gate_passes, _gate_reason, _text_length, _rank }

Usage and CLI Flags:
    python preprocess_reviews.py                # all discovered products
    python preprocess_reviews.py --only <slug>  # one product
'''

import argparse
import json
import os
import statistics
import sys

from collections import Counter


# Try to prevent encoding errors.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

## CONSTANTS ##
DATA_DIR = 'data'
FILTER_CACHE_DIR = 'data/filter_caches'
PROMPT_VERSION = 'v1'

MIN_TEXT_LEN = 50
MAX_TEXT_LEN = 3000


def cache_path_for(slug: str) -> str:
    '''Canonical path for a product's filter-decision cache. Must match
    build_filter_cache.py's scheme.'''
    return os.path.join(FILTER_CACHE_DIR, f'{slug}_filter_decisions_{PROMPT_VERSION}.json')


def discover_products() -> dict[str, str]:
    '''Return {slug: dir} for each data/*/ with reviews.jsonl, summary.json,
    AND a built filter cache at data/filter_caches/{slug}_filter_decisions_v1.json.'''
    out: dict[str, str] = {}
    if not os.path.isdir(DATA_DIR):
        return out
    for name in sorted(os.listdir(DATA_DIR)):
        pdir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(pdir) or name == 'filter_caches':
            continue
        r = os.path.join(pdir, 'reviews.jsonl')
        s = os.path.join(pdir, 'summary.json')
        if not (os.path.exists(r) and os.path.getsize(r) > 0 and os.path.exists(s)):
            continue
        if not os.path.exists(cache_path_for(name)):
            continue  # cache not built yet
        out[name] = pdir
    return out


def resolve_only(products: dict[str, str], needle: str) -> dict[str, str]:
    matches = [k for k in products if needle.lower() in k.lower()]
    if len(matches) == 0:
        print(f'[!] --only {needle!r} matches no product. Available: {sorted(products)}')
        sys.exit(1)
    if len(matches) > 1:
        print(f'[!] --only {needle!r} matches multiple: {matches}.')
        sys.exit(1)
    return {matches[0]: products[matches[0]]}


def synth_id(r: dict) -> str:
    '''Stable review id — must match build_filter_cache.py's scheme.'''
    return f'{r.get("user_id","unk")}_{r.get("timestamp",0)}'


def review_text(r: dict) -> str:
    '''Combined title+text used for both length hygiene and the gate.'''
    return ((r.get('title') or '').strip() + ' ' + (r.get('text') or '').strip()).strip()


def load_raw(path: str) -> list[dict]:
    reviews = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                reviews.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return reviews


def load_cache(path: str) -> dict[str, dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f).get('decisions', {})


## PER-PRODUCT PIPELINE ##

def process(product_key: str, product_dir: str) -> dict:
    raw_path = os.path.join(product_dir, 'reviews.jsonl')
    cache_path = cache_path_for(product_key)
    out_path = os.path.join(product_dir, 'reviews_ranked.jsonl')

    if not os.path.exists(raw_path) or not os.path.exists(cache_path):
        print(f'  [{product_key}] missing inputs; skipping')
        return {'product': product_key, 'skipped': True}

    raw = load_raw(raw_path)
    cache = load_cache(cache_path)

    # Dedupe by (user_id, timestamp). Verified same text in Amazon data.
    seen: set[str] = set()
    dedupe_drops = 0
    deduped: list[dict] = []
    for r in raw:
        rid = synth_id(r)
        if rid in seen:
            dedupe_drops += 1
            continue
        seen.add(rid)
        deduped.append(r)

    # Hygiene
    hygiene_kept: list[tuple[dict, str, int]] = []  # (review, text, length)
    hygiene_empty = 0
    hygiene_too_short = 0
    hygiene_too_long = 0
    for r in deduped:
        text = review_text(r)
        n = len(text)
        if n == 0:
            hygiene_empty += 1
            continue
        if n < MIN_TEXT_LEN:
            hygiene_too_short += 1
            continue
        if n > MAX_TEXT_LEN:
            hygiene_too_long += 1
            continue
        hygiene_kept.append((r, text, n))

    # Gate
    gate_kept: list[tuple[dict, str, int, str]] = []  # (review, text, length, reason)
    gate_missing = 0   # id not found in cache (shouldn't happen)
    gate_errored = 0   # cache recorded passes_gate=null
    gate_failed = 0    # passes_gate=false
    for r, text, n in hygiene_kept:
        rid = synth_id(r)
        d = cache.get(rid)
        if d is None:
            gate_missing += 1
            continue
        pg = d.get('passes_gate')
        if pg is None:
            gate_errored += 1
            continue
        if pg is False:
            gate_failed += 1
            continue
        gate_kept.append((r, text, n, d.get('reason', '')))

    # Sort: helpful_vote DESC, timestamp DESC.
    gate_kept.sort(
        key=lambda rec: (rec[0].get('helpful_vote', 0) or 0,
                         rec[0].get('timestamp', 0) or 0),
        reverse=True,
    )

    # Write ranked jsonl with added fields.
    os.makedirs(product_dir, exist_ok=True)
    tmp = out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for rank, (r, text, n, reason) in enumerate(gate_kept, 1):
            out = dict(r)
            out['_gate_passes'] = True
            out['_gate_reason'] = reason
            out['_text_length'] = n
            out['_rank'] = rank
            f.write(json.dumps(out, ensure_ascii=False) + '\n')
    os.replace(tmp, out_path)

    # Report
    print(f'\n[{product_key}] {out_path}')
    print(f'  raw              : {len(raw):>6,}')
    print(f'  after dedupe     : {len(deduped):>6,}   (-{dedupe_drops:>4,} duplicate ids)')
    hyg_out = len(hygiene_kept)
    print(f'  after hygiene    : {hyg_out:>6,}   '
          f'(-{hygiene_empty:>4,} empty, -{hygiene_too_short:,} <{MIN_TEXT_LEN}ch, '
          f'-{hygiene_too_long} >{MAX_TEXT_LEN}ch)')
    gate_out = len(gate_kept)
    gate_drop = hyg_out - gate_out
    extra = []
    if gate_missing:
        extra.append(f'{gate_missing} missing from cache')
    if gate_errored:
        extra.append(f'{gate_errored} LLM errors')
    extra_str = f" + {', '.join(extra)}" if extra else ''
    print(f'  after gate       : {gate_out:>6,}   '
          f'(-{gate_drop:,}: {gate_failed} failed gate{extra_str})')
    print(f'  FINAL            : {gate_out:>6,}')

    # Rating distribution of final set
    ratings = Counter(int(r.get('rating', 0)) for r, _, _, _ in gate_kept)
    rating_parts = [f'{s}\u2605 {ratings.get(s, 0):,}' for s in (5, 4, 3, 2, 1)]
    print(f'  rating dist      : ' + '  '.join(rating_parts))

    # Helpful-vote stats
    hvs = [(r.get('helpful_vote', 0) or 0) for r, _, _, _ in gate_kept]
    if hvs:
        print(f'  helpful_votes    : median={statistics.median(hvs):.0f}  '
              f'max={max(hvs):,}  mean={statistics.mean(hvs):.2f}')

    # Top 5 and Bottom 5
    def brief(r, rank):
        title = (r.get('title') or '').replace('\n', ' ').strip()
        if len(title) > 70:
            title = title[:67] + '...'
        return (f'    [#{rank:>4}] rating={int(r.get("rating",0))}  '
                f'helpful={r.get("helpful_vote",0):>3}  {title}')

    print('  top 5 ranked:')
    for rank, (r, _, _, _) in enumerate(gate_kept[:5], 1):
        print(brief(r, rank))
    if len(gate_kept) > 5:
        print('  bottom 5 ranked:')
        start = len(gate_kept) - 5
        for i, (r, _, _, _) in enumerate(gate_kept[-5:], 1):
            print(brief(r, start + i))

    return {
        'product': product_key,
        'raw': len(raw),
        'deduped': len(deduped),
        'hygiene_kept': hyg_out,
        'final': gate_out,
        'gate_missing': gate_missing,
        'gate_errored': gate_errored,
        'gate_failed': gate_failed,
    }

## MAIN ##

def main():
    parser = argparse.ArgumentParser(description='Build reviews_ranked.jsonl per product.')
    parser.add_argument('--only', default=None,
                        help='Substring-match a single product slug (e.g. "chess", "backpack").')
    args = parser.parse_args()

    products = discover_products()
    if not products:
        print(f'[!] No products found under {DATA_DIR}/ with a built cache.')
        sys.exit(1)
    print(f'Discovered products (with cache): {sorted(products)}')

    selected = resolve_only(products, args.only) if args.only else products

    results = []
    for key, pdir in selected.items():
        results.append(process(key, pdir))

    # Aggregate line
    print('\n' + '=' * 72)
    print('  AGGREGATE')
    print('=' * 72)
    for s in results:
        if s.get('skipped'):
            continue
        print(f'  {s["product"]:<6}  raw={s["raw"]:>6,}  '
              f'deduped={s["deduped"]:>6,}  '
              f'hygiene={s["hygiene_kept"]:>6,}  '
              f'final={s["final"]:>6,}')


if __name__ == '__main__':
    main()
