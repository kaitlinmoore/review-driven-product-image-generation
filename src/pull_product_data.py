'''pull_product_data.py
--------------------
One-time data pull for the six candidate products explored during product
selection: three canonical (water_bottle, chess_set, jeans) plus three
retained as methodological evidence of the candidate-exploration process
(backpack, headphones, espresso_machine).

For each product, this script:
    1. Streams the category's meta file from UCSD and extracts the full
       metadata record for that parent_asin (title, description, features,
       all image URLs, brand, price, etc.).
    2. Streams the category's review file from UCSD and saves every review
       matching that parent_asin to a local .jsonl file.
    3. Downloads all product images (MAIN + alternate angles) from Amazon's
       CDN to a local folder.
    4. Writes a summary JSON with aggregate stats (review count, avg rating,
       helpful-vote distribution, etc.).

The script writes reviews to disk incrementally as they are found.
If the script is interrupted mid-stream, you can re-run it and it
will skip products that already completed.

Usage:
    python pull_product_data.py                    # Pull all six products.
    python pull_product_data.py --only <slug>      # Pull only specific product.
    python pull_product_data.py --force            # Re-pull even if output exists.

Output structure:
    data/
        water_bottle/
            metadata.json
            reviews.jsonl
            summary.json
            images/
                main.jpg
                alt_01.jpg
                ...
        chess_set/
            ...
        jeans/
            ...
        backpack/                # candidate-selection evidence
            ...
        headphones/              # candidate-selection evidence
            ...
        espresso_machine/        # candidate-selection evidence
            ...
'''

import argparse
import gzip
import io
import json
import requests
import sys
import time

from dataclasses import dataclass
from pathlib import Path


## PRODUCT CONFIG ##
# To swap products later, edit only these entries. Everything else downstream
# will automatically key off parent_asin and category.

@dataclass
class ProductTarget:
    slug: str            # short folder name (e.g. 'water_bottle')
    display_name: str    # human-readable name for log output
    parent_asin: str     # the one field that uniquely identifies the listing
    category: str        # must match McAuley's category slug exactly
    variant_note: str    # free-text: which variant is our 'canonical' ground truth

# Defined all final candidate products. 3 will be used for canonical run.
PRODUCTS = [
    ProductTarget(
        slug='water_bottle',
        display_name='HYDRO CELL Fuchsia 24oz Sports Water Bottle',
        parent_asin='B09GBG54BY',
        category='Home_and_Kitchen',
        variant_note='Canonical variant: Fuchsia 24oz (center of composite hero image)',
    ),
    ProductTarget(
        slug='chess_set',
        display_name='Wegiel Handmade European Ambassador Chess Set',
        parent_asin='B0009WSPRO',
        category='Toys_and_Games',
        variant_note='Canonical variant: natural wood (3 wood-stain variants exist; pick based on review prominence later)',
    ),
    ProductTarget(
        slug='jeans',
        display_name="WallFlower Women's Legendary Bootcut Jeans",  # double-quoted: contains apostrophe
        parent_asin='B0964KXQNG',
        category='Clothing_Shoes_and_Jewelry',
        variant_note='Canonical variant: medium-wash bootcut as shown in hero image',
    ),
    ProductTarget(
        slug='backpack',
        display_name='Amazon Basics Internal Frame Hiking Backpack with Rainfly',
        parent_asin='B077P17P2N',
        category='Sports_and_Outdoors',
        variant_note='Canonical variant: Black, 75L internal-frame (confirmed from metadata: Color=Black, Size=75 L). Amazon Basics hiking backpacks also ship in 55L/65L under sibling ASINs; this parent_asin covers the 75L SKU.',
    ),
    ProductTarget(
        slug='headphones',
        display_name='Beats Studio3 Wireless Noise Cancelling Over-Ear Headphones (Blue)',
        parent_asin='B07Q39HCK1',
        category='Electronics',
        variant_note='Canonical variant: Blue, Previous Model (confirmed from metadata: Color=Blue). Beats Studio3 shipped in several colors under sibling ASINs; this parent_asin covers the blue SKU.',
    ),
    ProductTarget(
        slug='espresso_machine',
        display_name='Breville Barista Express Espresso Machine (Brushed Stainless Steel, BES870XL)',
        parent_asin='B0B3D2KYNS',
        category='Home_and_Kitchen',
        variant_note='Canonical variant: Brushed Stainless Steel BES870XL (confirmed from metadata: Color=Brushed Stainless Steel; dimensions 12"D x 11"W x 13.5"H). Barista Express ships in other finishes (e.g. Cranberry Red, Sea Salt) under sibling ASINs; this parent_asin covers the brushed-stainless SKU.',
    ),
]


## URLS AND PATHS ##

UCSD_BASE = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw'

DATA_ROOT = Path('data')

# Browser-like headers so Amazon's CDN will serve the images.
# (It can be picky about User-Agent for unauthenticated requests.)
BROWSER_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.amazon.com/',
}


def meta_url(category: str) -> str:
    return f'{UCSD_BASE}/meta_categories/meta_{category}.jsonl.gz'


def review_url(category: str) -> str:
    return f'{UCSD_BASE}/review_categories/{category}.jsonl.gz'


def product_dir(product: ProductTarget) -> Path:
    return DATA_ROOT / product.slug


## FETCH PRODUCT METADATA ##

def fetch_metadata(product: ProductTarget) -> dict | None:
    '''
    Stream the meta_{Category}.jsonl.gz file and return the full record
    for this product's parent_asin. Returns None if not found.

    Meta files are small enough (tens to hundreds of MB compressed) that
    streaming through is fast.
    '''
    url = meta_url(product.category)
    print(f'  [meta] Streaming {url}')

    try:
        resp = requests.get(url, stream=True, timeout=60)
    except requests.RequestException as e:
        print(f'  [meta] Network error: {type(e).__name__}: {e}')
        return None

    if resp.status_code != 200:
        print(f'  [meta] HTTP {resp.status_code}')
        resp.close()
        return None

    scanned = 0
    t0 = time.time()
    target = product.parent_asin

    try:
        with gzip.GzipFile(fileobj=resp.raw) as gz:
            text = io.TextIOWrapper(gz, encoding='utf-8')
            for line in text:
                scanned += 1

                if scanned % 50_000 == 0:
                    elapsed = time.time() - t0
                    print(f'  [meta]   scanned {scanned:,} rows | {elapsed:.0f}s')

                # Cheap pre-filter: if ASIN isn't in the raw line, can't match.
                if target not in line:
                    continue

                try:
                    rec = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if rec.get('parent_asin') == target:
                    elapsed = time.time() - t0
                    print(f'  [meta]   FOUND after scanning {scanned:,} rows ({elapsed:.0f}s)')
                    return rec
    finally:
        resp.close()

    print(f'  [meta] parent_asin {target} NOT FOUND in {scanned:,} rows')
    return None


## STREAM ALL REVIEWS ##
# This will happen with incremental save.

def stream_and_save_reviews(product: ProductTarget, output_path: Path) -> dict:
    '''
    Stream the review file and write every matching review to output_path
    as jsonl (one JSON object per line). Returns an aggregate stats dict.

    Writes incrementally: if the stream fails halfway, the partial output
    is still on disk. Re-running with the same output_path will OVERWRITE
    what's there (this is a one-time pull, not an incremental sync).
    '''
    url = review_url(product.category)
    print(f'  [reviews] Streaming {url}')

    try:
        resp = requests.get(url, stream=True, timeout=60)
    except requests.RequestException as e:
        print(f'  [reviews] Network error: {type(e).__name__}: {e}')
        return {'error': str(e)}

    if resp.status_code != 200:
        print(f'  [reviews] HTTP {resp.status_code}')
        resp.close()
        return {'error': f'HTTP {resp.status_code}'}

    target = product.parent_asin
    scanned = 0
    matched = 0
    t0 = time.time()

    # Stats accumulators
    rating_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    helpful_vote_total = 0
    verified_count = 0
    has_text_count = 0
    min_timestamp = None
    max_timestamp = None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as out, \
             gzip.GzipFile(fileobj=resp.raw) as gz:
            text = io.TextIOWrapper(gz, encoding='utf-8')
            for line in text:
                scanned += 1

                if scanned % 250_000 == 0:
                    elapsed = time.time() - t0
                    rate = scanned / max(elapsed, 1)
                    print(f'  [reviews]   scanned {scanned:,} | '
                          f'matched {matched} | '
                          f'{elapsed:.0f}s ({rate:,.0f}/s)')

                # Cheap ASIN pre-filter before JSON parse
                if target not in line:
                    continue

                try:
                    rec = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if rec.get('parent_asin') != target:
                    continue

                # Write to disk immediately (incremental save).
                out.write(json.dumps(rec) + '\n')
                matched += 1

                # Accumulate stats.
                rating = rec.get('rating')
                if rating in rating_counts:
                    rating_counts[int(rating)] += 1

                helpful_vote_total += rec.get('helpful_vote', 0) or 0

                if rec.get('verified_purchase'):
                    verified_count += 1

                if (rec.get('text') or '').strip():
                    has_text_count += 1

                ts = rec.get('timestamp')
                if ts is not None:
                    if min_timestamp is None or ts < min_timestamp:
                        min_timestamp = ts
                    if max_timestamp is None or ts > max_timestamp:
                        max_timestamp = ts
    finally:
        resp.close()

    elapsed = time.time() - t0
    print(f'  [reviews] DONE. Scanned {scanned:,} rows in {elapsed:.0f}s. '
          f'Matched {matched} reviews.')

    # Compute average rating.
    total_rated = sum(rating_counts.values())
    avg_rating = (
        sum(r * n for r, n in rating_counts.items()) / total_rated
        if total_rated > 0 else None
    )

    return {
        'total_rows_scanned': scanned,
        'matched_reviews': matched,
        'reviews_with_text': has_text_count,
        'verified_purchase_count': verified_count,
        'rating_distribution': rating_counts,
        'avg_rating': avg_rating,
        'total_helpful_votes': helpful_vote_total,
        'min_timestamp': min_timestamp,
        'max_timestamp': max_timestamp,
        'stream_elapsed_seconds': round(elapsed, 1),
    }


## DOWNLOAD PRODUCT IMAGES ##

def extract_image_urls(metadata: dict) -> list[tuple[str, str]]:
    '''
    Pull all image URLs out of the metadata record. Returns a list of
    (label, url) pairs, e.g. [('main', 'https://...'), ('alt_01', 'https://...'), ...]

    McAuley Lab's meta records use an 'images' field which is a list of dicts,
    each with 'large', 'hi_res', 'thumb', and 'variant' keys. We prefer
    'hi_res' (or 'large' as fallback) for ground-truth purposes.
    '''
    images = metadata.get('images') or []
    if not isinstance(images, list):
        return []

    pairs = []
    main_taken = False

    for idx, img in enumerate(images):
        if not isinstance(img, dict):
            continue

        # Pick the highest-resolution URL available
        url = img.get('hi_res') or img.get('large') or img.get('thumb')
        if not url:
            continue

        variant = (img.get('variant') or '').strip()

        # The first MAIN variant becomes our 'main' image
        if variant.upper() == 'MAIN' and not main_taken:
            label = 'main'
            main_taken = True
        else:
            # Alternate images numbered sequentially
            label = f'alt_{len(pairs):02d}'

        pairs.append((label, url))

    # Guarantee we have at least a main if any images exist
    if pairs and not main_taken:
        pairs[0] = ('main', pairs[0][1])

    return pairs


def download_images(image_pairs: list[tuple[str, str]], images_dir: Path) -> dict:
    '''
    Download each image to images_dir/{label}.jpg. Returns a stats dict
    summarizing what succeeded and what failed.
    '''
    images_dir.mkdir(parents=True, exist_ok=True)

    results = {'downloaded': 0, 'failed': 0, 'errors': []}

    for label, url in image_pairs:
        # Derive extension from URL (default to .jpg which is what Amazon serves)
        extension = '.jpg'
        lower_url = url.lower()
        for ext in ('.jpg', '.jpeg', '.png', '.webp'):
            if ext in lower_url:
                extension = ext
                break

        output_path = images_dir / f'{label}{extension}'
        print(f'  [images]   {label}: {url[:80]}')

        try:
            r = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
            if r.status_code == 200 and len(r.content) > 100:
                output_path.write_bytes(r.content)
                results['downloaded'] += 1
                print(f'  [images]     -> saved {len(r.content):,} bytes')
            else:
                results['failed'] += 1
                results['errors'].append(f'{label}: HTTP {r.status_code}')
                print(f'  [images]     -> FAILED (HTTP {r.status_code})')
        except requests.RequestException as e:
            results['failed'] += 1
            results['errors'].append(f'{label}: {type(e).__name__}: {e}')
            print(f'  [images]     -> ERROR ({type(e).__name__})')

    return results


## PER-PRODUCT DRIVER ##

def pull_product(product: ProductTarget, force: bool = False) -> bool:
    '''
    Run the full pull (metadata + reviews + images) for one product.
    Returns True on success, False if any step failed catastrophically.
    '''
    print()
    print('=' * 80)
    print(f'  {product.display_name}')
    print(f'  parent_asin = {product.parent_asin}')
    print(f'  category    = {product.category}')
    print(f'  variant     = {product.variant_note}')
    print('=' * 80)

    pdir = product_dir(product)
    pdir.mkdir(parents=True, exist_ok=True)

    metadata_path = pdir / 'metadata.json'
    reviews_path = pdir / 'reviews.jsonl'
    summary_path = pdir / 'summary.json'
    images_dir = pdir / 'images'

    # Skip logic for re-runs
    if not force and metadata_path.exists() and reviews_path.exists() and summary_path.exists():
        print(f'  [skip] All output files already exist. Use --force to re-pull.')
        return True

    # Metadata
    print(f'\n  Step 1/3: Fetch product metadata')
    metadata = fetch_metadata(product)
    if metadata is None:
        print(f'  [!] Could not fetch metadata. Aborting this product.')
        return False

    # Save metadata as pretty-printed JSON for easy eyeball inspection.
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(f'  [meta] Saved to {metadata_path}')
    print(f'  [meta] Title: {(metadata.get("title") or "")[:100]}')
    print(f'  [meta] Store: {metadata.get("store")}')
    print(f'  [meta] Price: {metadata.get("price")}')
    print(f'  [meta] Rating: {metadata.get("average_rating")} '
          f'({metadata.get("rating_number")} ratings)')

    # Reviews
    print(f'\n  Step 2/3: Stream and save reviews')
    review_stats = stream_and_save_reviews(product, reviews_path)
    print(f'  [reviews] Saved to {reviews_path}')

    # Images
    print(f'\n  Step 3/3: Download product images')
    image_pairs = extract_image_urls(metadata)
    if not image_pairs:
        print(f'  [images] No image URLs found in metadata.')
        image_stats = {'downloaded': 0, 'failed': 0, 'errors': ['no urls in metadata']}
    else:
        print(f'  [images] Found {len(image_pairs)} image URLs')
        image_stats = download_images(image_pairs, images_dir)

    # Write summary.
    summary = {
        'product': {
            'slug': product.slug,
            'display_name': product.display_name,
            'parent_asin': product.parent_asin,
            'category': product.category,
            'variant_note': product.variant_note,
        },
        'metadata': {
            'title': metadata.get('title'),
            'store': metadata.get('store'),
            'price': metadata.get('price'),
            'main_category': metadata.get('main_category'),
            'average_rating': metadata.get('average_rating'),
            'rating_number': metadata.get('rating_number'),
            'description_length': sum(
                len(s) for s in (metadata.get('description') or [])
                if isinstance(s, str)
            ),
            'features_count': len(metadata.get('features') or []),
            'image_count': len(image_pairs),
        },
        'reviews': review_stats,
        'images': image_stats,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(f'\n  [summary] Saved to {summary_path}')

    return True


## MAIN ##

def main():
    parser = argparse.ArgumentParser(
        description='Pull all data (metadata, reviews, images) for the six '
                    'candidate products (three canonical plus three retained '
                    'as candidate-selection methodology evidence).'
    )
    parser.add_argument('--only', default=None,
                        help='Only pull products whose slug contains this substring '
                             '(e.g., "water", "chess", "jeans")')
    parser.add_argument('--force', action='store_true',
                        help='Re-pull even if output files already exist')
    args = parser.parse_args()

    targets = PRODUCTS
    if args.only:
        needle = args.only.lower()
        targets = [p for p in PRODUCTS if needle in p.slug.lower()
                   or needle in p.display_name.lower()]
        if not targets:
            print(f'[!] No products match "{args.only}". Available:')
            for p in PRODUCTS:
                print(f'    {p.slug}  ({p.display_name})')
            sys.exit(1)

    print(f'Pulling data for {len(targets)} product(s). Output root: {DATA_ROOT.resolve()}')

    t_start = time.time()
    results = []
    for product in targets:
        ok = pull_product(product, force=args.force)
        results.append((product, ok))

    # Final summary
    elapsed = time.time() - t_start
    print()
    print('=' * 80)
    print(f'  FINAL SUMMARY  ({elapsed:.0f}s total)')
    print('=' * 80)
    for product, ok in results:
        status = 'OK ' if ok else 'FAIL'
        print(f'  [{status}] {product.slug:15} {product.display_name}')

    if all(ok for _, ok in results):
        print(f'\n[DONE] All products pulled successfully.')
    else:
        print(f'\n[!] Some products failed. Check logs above.')
        sys.exit(1)


if __name__ == '__main__':
    main()
