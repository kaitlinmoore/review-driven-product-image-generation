'''explore_candidates.py
---------------------
Stream the McAuley Lab Amazon Reviews 2023 metadata for a specific category
and produce a ranked list of candidate products WITHOUT downloading the
full file.

This version reads directly from the UCSD data repository
(datarepo.eng.ucsd.edu), which hosts gzipped JSONL files for every category.
This bypasses:
  - The HuggingFace datasets library (rejects loading scripts)
  - HuggingFace Parquet files (only covers 9 of 33 categories so far)

Mechanism: stream the .gz file via HTTP, decompress on the fly, parse
line-by-line, filter, stop when target count reached.

Usage:
    python explore_candidates.py --category Home_and_Kitchen --keyword "water bottle" --popularity high
    python explore_candidates.py --list-categories
'''

import argparse
import csv
import gzip
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path

import requests


## CONFIGURATION ##

POPULARITY_BANDS = {
    'mega':   (5000, float('inf')),   # Bestsellers (5K+ ratings)
    'high':   (1000, 5000),           # Established popular (1K-5K)
    'mid':    (100, 1000),            # Established (100-1K)
    'niche':  (20, 100),              # Long-tail (20-100)
    'any':    (0, float('inf')),      # No filter
}

DEFAULT_TARGET_COUNT = 30
MAX_ROWS_TO_SCAN = 200_000

UCSD_BASE = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw'

# All 33 categories in the dataset, per the official documentation
ALL_CATEGORIES = [
    'All_Beauty',
    'Amazon_Fashion',
    'Appliances',
    'Arts_Crafts_and_Sewing',
    'Automotive',
    'Baby_Products',
    'Beauty_and_Personal_Care',
    'Books',
    'CDs_and_Vinyl',
    'Cell_Phones_and_Accessories',
    'Clothing_Shoes_and_Jewelry',
    'Digital_Music',
    'Electronics',
    'Gift_Cards',
    'Grocery_and_Gourmet_Food',
    'Handmade_Products',
    'Health_and_Household',
    'Health_and_Personal_Care',
    'Home_and_Kitchen',
    'Industrial_and_Scientific',
    'Kindle_Store',
    'Magazine_Subscriptions',
    'Movies_and_TV',
    'Musical_Instruments',
    'Office_Products',
    'Patio_Lawn_and_Garden',
    'Pet_Supplies',
    'Software',
    'Sports_and_Outdoors',
    'Subscription_Boxes',
    'Tools_and_Home_Improvement',
    'Toys_and_Games',
    'Unknown',
    'Video_Games',
]


def metadata_url(category: str) -> str:
    '''Construct the URL for a category's metadata jsonl.gz file.'''
    return f'{UCSD_BASE}/meta_categories/meta_{category}.jsonl.gz'


## STREAMING JSONL.GZ OVER HTTP ##

def stream_metadata_rows(category: str, timeout: int = 60):
    '''Generator that yields one metadata record dict at a time.
    Streams the gzipped JSONL file over HTTP without loading it all into memory.'''
    url = metadata_url(category)
    print(f'[*] Streaming: {url}')

    try:
        # stream=True means content is downloaded in chunks as we iterate,
        # not buffered all at once
        resp = requests.get(url, stream=True, timeout=timeout)
    except requests.RequestException as e:
        print(f'[!] Network error: {type(e).__name__}: {e}')
        sys.exit(1)

    if resp.status_code == 404:
        print(f'[!] File not found (HTTP 404). Check category spelling.')
        print(f'    Run --list-categories to see valid options.')
        sys.exit(1)
    elif resp.status_code != 200:
        print(f'[!] HTTP {resp.status_code} from UCSD server.')
        sys.exit(1)

    # Show the file size if the server tells us
    total = resp.headers.get('Content-Length')
    if total:
        total_mb = int(total) / (1024 * 1024)
        print(f'[*] Compressed file size: {total_mb:.1f} MB')
        print(f'[*] (we stream and filter -- only the rows we need get parsed)')

    # Wrap the raw HTTP stream in a gzip decoder, then in a text reader
    try:
        with gzip.GzipFile(fileobj=resp.raw) as gz:
            text_stream = io.TextIOWrapper(gz, encoding='utf-8')
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    finally:
        resp.close()


## FILTERING ##

def matches_keyword(title, keyword: str) -> bool:
    '''Case-insensitive keyword match against the product title only.
    Searching title-only avoids false positives from products that merely
    mention the keyword in their description (e.g., a milk frothing pitcher
    that says "for espresso machines" in its description).'''
    if not keyword:
        return True
    return keyword.lower() in (title or '').lower()


def has_usable_images(images) -> bool:
    '''Meta format: images is a LIST of dicts, each dict has hi_res/large/thumb/variant.
    (Different from the Parquet format which flattened it into a dict of lists.)'''
    if not images:
        return False
    if isinstance(images, list):
        for img in images:
            if isinstance(img, dict):
                for key in ('hi_res', 'large'):
                    if img.get(key):
                        return True
    elif isinstance(images, dict):
        # In case a future format change makes it a dict-of-lists
        for key in ('hi_res', 'large'):
            urls = images.get(key, []) or []
            for u in urls:
                if u:
                    return True
    return False


def get_main_image_url(images) -> str:
    '''Return the MAIN variant URL, falling back to first available.'''
    if not images:
        return ''

    # jsonl format: list of dicts
    if isinstance(images, list):
        # Prefer the MAIN variant
        for img in images:
            if isinstance(img, dict) and img.get('variant') == 'MAIN':
                return img.get('hi_res') or img.get('large') or ''
        # Fallback: first usable URL
        for img in images:
            if isinstance(img, dict):
                url = img.get('hi_res') or img.get('large') or ''
                if url:
                    return url
        return ''

    # parquet-style format: dict of parallel lists
    if isinstance(images, dict):
        variants = images.get('variant', []) or []
        hi_res = images.get('hi_res', []) or []
        large = images.get('large', []) or []
        for i, v in enumerate(variants):
            if v == 'MAIN':
                if i < len(hi_res) and hi_res[i]:
                    return hi_res[i]
                if i < len(large) and large[i]:
                    return large[i]
        for url in list(hi_res) + list(large):
            if url:
                return url
    return ''


def collect_candidates(category, popularity, keyword='',
                        target_count=DEFAULT_TARGET_COUNT,
                        max_rows=MAX_ROWS_TO_SCAN):
    low, high = POPULARITY_BANDS[popularity]
    print(f'[*] Target: {target_count} candidates')
    print(f"[*] Popularity band '{popularity}': {low} <= rating_number < {high}")
    if keyword:
        print(f"[*] Keyword filter: '{keyword}'")
    print()

    candidates = []
    scanned = 0
    t_start = time.time()

    for record in stream_metadata_rows(category):
        scanned += 1

        if scanned % 10_000 == 0:
            elapsed = time.time() - t_start
            print(f'    scanned {scanned:,} rows | '
                  f'candidates found: {len(candidates)} | '
                  f'elapsed: {elapsed:.0f}s')

        rating_number = record.get('rating_number') or 0
        if not (low <= rating_number < high):
            continue

        if not has_usable_images(record.get('images')):
            continue

        title = record.get('title') or ''
        if not matches_keyword(title, keyword):
            continue

        desc = record.get('description') or []
        feats = record.get('features') or []
        desc_text = ' '.join(str(x) for x in desc) if isinstance(desc, list) else str(desc)
        feat_text = ' '.join(str(x) for x in feats) if isinstance(feats, list) else str(feats)
        if len(desc_text) + len(feat_text) < 50:
            continue

        candidates.append({
            'parent_asin':    record.get('parent_asin'),
            'title':          title,
            'rating_number':  rating_number,
            'average_rating': record.get('average_rating'),
            'main_category':  record.get('main_category'),
            'store':          record.get('store'),
            'price':          record.get('price'),
            'main_image_url': get_main_image_url(record.get('images')),
            'description_preview': (desc_text[:200] + '...') if len(desc_text) > 200 else desc_text,
        })

        if len(candidates) >= target_count:
            print(f'\n[OK] Reached target of {target_count} candidates. Stopping scan.')
            break

        if scanned >= max_rows:
            print(f'\n[!] Reached max scan limit of {max_rows:,} rows. Stopping.')
            break

    elapsed = time.time() - t_start
    print(f'\n[OK] Scanned {scanned:,} rows in {elapsed:.1f}s. '
          f'Found {len(candidates)} candidates.')
    return candidates


## IMAGE URL VALIDATION ##

def validate_image_url(url: str, timeout: int = 10):
    if not url:
        return False, 'no URL'
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200:
            ctype = resp.headers.get('Content-Type', '')
            if 'image' in ctype.lower():
                return True, f'OK ({ctype})'
            return False, f'not an image ({ctype})'
        return False, f'HTTP {resp.status_code}'
    except requests.RequestException as e:
        return False, f'error: {type(e).__name__}'


def validate_candidates(candidates, check_top_n=10):
    print(f'\n[*] Validating image URLs for top {check_top_n} candidates...')
    candidates_sorted = sorted(candidates, key=lambda c: c['rating_number'], reverse=True)
    for i, cand in enumerate(candidates_sorted[:check_top_n]):
        is_valid, note = validate_image_url(cand['main_image_url'])
        cand['image_valid'] = is_valid
        cand['image_note'] = note
        status = 'OK ' if is_valid else 'X  '
        print(f'    [{i+1:2d}] {status} {cand["parent_asin"]} -- {note}')
    for cand in candidates_sorted[check_top_n:]:
        cand['image_valid'] = None
        cand['image_note'] = 'not checked'
    return candidates_sorted


## OUTPUT ##

def write_csv(candidates, output_path: Path):
    fields = [
        'parent_asin', 'title', 'rating_number', 'average_rating',
        'main_category', 'store', 'price', 'main_image_url',
        'image_valid', 'image_note', 'description_preview',
    ]
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for cand in candidates:
            writer.writerow(cand)
    print(f'\n[OK] Wrote {len(candidates)} candidates to {output_path}')


def print_top_candidates(candidates, n=5):
    print('\n' + '=' * 80)
    print(f'TOP {n} CANDIDATES')
    print('=' * 80)
    for i, cand in enumerate(candidates[:n], 1):
        valid_str = ''
        if cand.get('image_valid') is True:
            valid_str = '   [image: OK]'
        elif cand.get('image_valid') is False:
            valid_str = f'   [image: FAIL {cand.get("image_note", "")}]'
        print(f'\n{i}. {cand["title"][:80]}{valid_str}')
        print(f'   parent_asin: {cand["parent_asin"]}')
        print(f'   rating_number: {cand["rating_number"]:,} | '
              f'avg: {cand.get("average_rating", "N/A")} | '
              f'store: {cand.get("store", "N/A")}')


## DIAGNOSTICS ##

def list_categories():
    '''List the 33 valid category names.'''
    print('All categories in the McAuley Amazon Reviews 2023 dataset:\n')
    for c in ALL_CATEGORIES:
        print(f'    {c}')
    print(f'\nTotal: {len(ALL_CATEGORIES)} categories')
    print(f'\nUsage example:')
    print(f'    python explore_candidates.py --category Home_and_Kitchen '
          f'--keyword "water bottle" --popularity high')


## CLI ##

def main():
    parser = argparse.ArgumentParser(
        description='Find candidate products in McAuley Amazon Reviews 2023 '
                    'via direct UCSD jsonl.gz streaming.'
    )
    parser.add_argument('--category', default=None)
    parser.add_argument('--popularity', default='high',
                        choices=list(POPULARITY_BANDS.keys()))
    parser.add_argument('--keyword', default='')
    parser.add_argument('--target', type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument('--max-rows', type=int, default=MAX_ROWS_TO_SCAN)
    parser.add_argument('--validate-top', type=int, default=10)
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--list-categories', action='store_true',
                        help='List all valid category names and exit')
    args = parser.parse_args()

    if args.list_categories:
        list_categories()
        return

    if not args.category:
        print('[!] --category is required (unless using --list-categories)')
        parser.print_help()
        sys.exit(1)

    if args.category not in ALL_CATEGORIES:
        print(f"[!] '{args.category}' is not a recognized category.")
        print(f'    Run --list-categories to see valid options.')
        # Try case-insensitive match to suggest a correction
        for c in ALL_CATEGORIES:
            if c.lower() == args.category.lower():
                print(f"    Did you mean '{c}'? (Categories are case-sensitive.)")
                break
        sys.exit(1)

    candidates = collect_candidates(
        category=args.category,
        popularity=args.popularity,
        keyword=args.keyword,
        target_count=args.target,
        max_rows=args.max_rows,
    )

    if not candidates:
        print('\n[!] No candidates found. Try loosening filters.')
        sys.exit(1)

    candidates = validate_candidates(candidates, check_top_n=args.validate_top)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keyword_suffix = (
        f'_{re.sub(r"[^a-z0-9]+", "_", args.keyword.lower())}'
        if args.keyword else ''
    )
    output_path = out_dir / f'candidates_{args.category}_{args.popularity}{keyword_suffix}.csv'
    write_csv(candidates, output_path)

    print_top_candidates(candidates, n=min(5, len(candidates)))


if __name__ == '__main__':
    main()
