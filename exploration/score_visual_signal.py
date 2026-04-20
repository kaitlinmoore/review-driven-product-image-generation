'''score_visual_signal.py
----------------------
For each candidate product (identified by parent_asin and category), stream
the review file from UCSD, filter to that product's reviews, and score how
much 'visual signal' the reviews contain.

Visual signal = fraction of reviews that describe the product's appearance
using words about color, material, shape, finish, size, or visual condition.

A review scores 'high' if it contains 3+ distinct visual descriptor words.
A product's final score is the percentage of sampled reviews that score high.

Usage:
    python score_visual_signal.py                    # score the default 7 candidates
    python score_visual_signal.py --sample-size 300  # scan more reviews per product
'''

import argparse
import gzip
import io
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import requests

from visual_vocabulary import visual_word_set


## CANDIDATE LIST ##

@dataclass
class Candidate:
    name: str
    parent_asin: str
    category: str
    title: str  # short version for display


CANDIDATES = [
    Candidate(
        name='Water bottle',
        parent_asin='B09L7RZ96X',
        category='Home_and_Kitchen',
        title='Simple Modern 64oz Summit Water Bottle',
    ),
    Candidate(
        name='Chess set',
        parent_asin='B09JZNWFFG',
        category='Toys_and_Games',
        title='Chess Armory 15 Inch Wooden Chess Set',
    ),
    Candidate(
        name='Board game (Sorry!)',
        parent_asin='B00000IWD0',
        category='Toys_and_Games',
        title='Sorry! Classic Hasbro Board Game',
    ),
    Candidate(
        name="Jeans (Levi's)",
        parent_asin='B08KR19G7S',
        category='Clothing_Shoes_and_Jewelry',
        title="Levi's Women's Mile High Super Skinny Jeans",
    ),
    Candidate(
        name='Stand mixer (KitchenAid)',
        parent_asin='B0C34HVTJN',
        category='Home_and_Kitchen',
        title='KitchenAid Artisan 5-Quart Stand Mixer',
    ),
    Candidate(
        name='Espresso machine (Breville)',
        parent_asin='B0B3D2KYNS',
        category='Home_and_Kitchen',
        title='Breville Barista Express BES870XL',
    ),
    Candidate(
        name='Hiking backpack (Venture Pal)',
        parent_asin='B08JD12G4X',
        category='Sports_and_Outdoors',
        title='Venture Pal 40L Hiking Backpack',
    ),
]


## VISUAL DESCRIPTOR VOCABULARY ##
# The vocabulary itself now lives in `visual_vocabulary.py` (shared with the
# preprocessing pipeline). `visual_word_set` is imported at the top of this
# file. Only the threshold is local — this script keeps its stricter bar of 3
# because it scores candidate richness, not per-review pass/fail gating.

# Threshold: this many distinct visual words = a 'visually descriptive' review.
# 3 is a deliberately moderate bar — strong enough to filter noise, low enough
# to catch short-but-visual reviews like 'Silver chrome finish is beautiful.'
VISUAL_DESCRIPTOR_THRESHOLD = 3


## SCORING ##


@dataclass
class ReviewScore:
    text: str
    rating: float
    helpful_vote: int
    distinct_visual_words: int
    visual_words: set = field(default_factory=set)

    @property
    def is_visual(self) -> bool:
        return self.distinct_visual_words >= VISUAL_DESCRIPTOR_THRESHOLD


@dataclass
class ProductScoreCard:
    candidate: Candidate
    reviews_sampled: int = 0
    reviews_with_signal: int = 0
    total_distinct_visual_words: int = 0
    top_visual_words: Counter = field(default_factory=Counter)
    exemplar_high_signal: list = field(default_factory=list)
    exemplar_low_signal: list = field(default_factory=list)

    @property
    def pct_visual(self) -> float:
        if self.reviews_sampled == 0:
            return 0.0
        return 100.0 * self.reviews_with_signal / self.reviews_sampled

    @property
    def avg_visual_words(self) -> float:
        if self.reviews_sampled == 0:
            return 0.0
        return self.total_distinct_visual_words / self.reviews_sampled


## REVIEW FILE STREAMING ##

UCSD_BASE = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw'


def review_url(category: str) -> str:
    return f'{UCSD_BASE}/review_categories/{category}.jsonl.gz'


def stream_reviews_for_product(category: str, parent_asin: str,
                                sample_size: int,
                                max_rows: int = 5_000_000):
    '''Stream the category's review file and yield reviews that match parent_asin.
    Stop once we have `sample_size` matches or we've scanned `max_rows` total.'''
    url = review_url(category)
    print(f'    Streaming reviews from {url}')

    try:
        resp = requests.get(url, stream=True, timeout=60)
    except requests.RequestException as e:
        print(f'    [!] Network error: {type(e).__name__}: {e}')
        return

    if resp.status_code != 200:
        print(f'    [!] HTTP {resp.status_code}')
        resp.close()
        return

    matched = 0
    scanned = 0
    t0 = time.time()

    try:
        with gzip.GzipFile(fileobj=resp.raw) as gz:
            text = io.TextIOWrapper(gz, encoding='utf-8')
            for line in text:
                scanned += 1

                if scanned % 100_000 == 0:
                    elapsed = time.time() - t0
                    print(f'      scanned {scanned:,} reviews | '
                          f'matched {matched} | {elapsed:.0f}s')

                if parent_asin not in line:
                    # Cheap pre-filter before JSON parsing. If the parent_asin
                    # string isn't anywhere in this line, it can't match.
                    continue

                try:
                    rec = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if rec.get('parent_asin') != parent_asin:
                    continue

                matched += 1
                yield rec

                if matched >= sample_size:
                    return
                if scanned >= max_rows:
                    print(f'    [!] Hit max_rows={max_rows:,}. Stopping scan.')
                    return
    finally:
        resp.close()


## PER-PRODUCT SCORING DRIVER ##

def score_product(candidate: Candidate, sample_size: int) -> ProductScoreCard:
    '''Score a single product candidate.'''
    print(f'\n[*] Scoring: {candidate.name} ({candidate.parent_asin})')
    print(f'    Title: {candidate.title}')

    card = ProductScoreCard(candidate=candidate)

    for rec in stream_reviews_for_product(candidate.category,
                                            candidate.parent_asin,
                                            sample_size):
        text = (rec.get('text') or '').strip()
        title = (rec.get('title') or '').strip()
        combined = title + ' ' + text

        if len(combined.strip()) < 20:
            # Skip trivial reviews entirely
            continue

        visual_words = visual_word_set(combined)
        score = ReviewScore(
            text=combined,
            rating=rec.get('rating', 0) or 0,
            helpful_vote=rec.get('helpful_vote', 0) or 0,
            distinct_visual_words=len(visual_words),
            visual_words=visual_words,
        )

        card.reviews_sampled += 1
        card.total_distinct_visual_words += len(visual_words)
        card.top_visual_words.update(visual_words)

        if score.is_visual:
            card.reviews_with_signal += 1
            # Save top exemplars (prefer helpful_vote as tiebreaker)
            if len(card.exemplar_high_signal) < 3:
                card.exemplar_high_signal.append(score)
            elif score.helpful_vote > min(e.helpful_vote for e in card.exemplar_high_signal):
                # Replace the least-helpful exemplar
                card.exemplar_high_signal.sort(key=lambda e: e.helpful_vote)
                card.exemplar_high_signal[0] = score
        else:
            # Low-signal exemplars, for contrast
            if len(card.exemplar_low_signal) < 2 and len(combined) > 50:
                card.exemplar_low_signal.append(score)

    print(f'    Sampled {card.reviews_sampled} reviews | '
          f'{card.reviews_with_signal} visually-rich '
          f'({card.pct_visual:.1f}%)')
    return card


## OUTPUT ##

def print_scorecard(card: ProductScoreCard):
    '''Pretty-print one product's scorecard.'''
    print()
    print('=' * 80)
    print(f'  {card.candidate.name}')
    print(f'  {card.candidate.title}')
    print('=' * 80)
    print(f'  parent_asin:       {card.candidate.parent_asin}')
    print(f'  category:          {card.candidate.category}')
    print(f'  reviews sampled:   {card.reviews_sampled}')
    print(f'  visually rich:     {card.reviews_with_signal}  ({card.pct_visual:.1f}%)')
    print(f'  avg visual words:  {card.avg_visual_words:.2f} distinct per review')

    if card.top_visual_words:
        top = card.top_visual_words.most_common(10)
        formatted = ', '.join(f'{w} ({n})' for w, n in top)
        print(f'  most common:       {formatted}')

    if card.exemplar_high_signal:
        print(f'\n  --- EXAMPLE HIGH-SIGNAL REVIEWS ---')
        for i, ex in enumerate(card.exemplar_high_signal[:2], 1):
            snippet = ex.text[:280] + ('...' if len(ex.text) > 280 else '')
            print(f'  [{i}] ({ex.distinct_visual_words} visual words, '
                  f'{ex.helpful_vote} helpful votes)')
            print(f'      {snippet}')
            print()

    if card.exemplar_low_signal:
        print(f'  --- EXAMPLE LOW-SIGNAL REVIEW (for contrast) ---')
        ex = card.exemplar_low_signal[0]
        snippet = ex.text[:200] + ('...' if len(ex.text) > 200 else '')
        print(f'  ({ex.distinct_visual_words} visual words)')
        print(f'      {snippet}')


def print_summary_table(cards: list):
    '''Print a side-by-side comparison table of all products.'''
    print('\n\n')
    print('=' * 90)
    print('  SUMMARY: VISUAL SIGNAL RANKING')
    print('=' * 90)
    print(f'  {"Product":<30} {"Sampled":>8} {"Visual":>8} {"% Visual":>10} {"Avg words":>10}')
    print(f'  {"-"*30} {"-"*8} {"-"*8} {"-"*10} {"-"*10}')

    # Sort by pct_visual desc
    for card in sorted(cards, key=lambda c: c.pct_visual, reverse=True):
        print(f'  {card.candidate.name:<30} '
              f'{card.reviews_sampled:>8} '
              f'{card.reviews_with_signal:>8} '
              f'{card.pct_visual:>9.1f}% '
              f'{card.avg_visual_words:>10.2f}')

    print('\n  Higher % visual and higher avg words = more visual signal in reviews,')
    print('  which means Phase 2 extraction has more to work with.')


## CLI ##

def main():
    parser = argparse.ArgumentParser(
        description='Score the visual signal richness of reviews for each '
                    'candidate product.'
    )
    parser.add_argument('--sample-size', type=int, default=200,
                        help='Max reviews to sample per product (default: 200)')
    parser.add_argument('--max-rows', type=int, default=10_000_000,
                        help='Safety cap on total rows scanned per category')
    parser.add_argument('--only', default=None,
                        help="Only score this candidate name substring "
                             "(e.g., 'chess' or 'water')")
    args = parser.parse_args()

    candidates_to_score = CANDIDATES
    if args.only:
        needle = args.only.lower()
        candidates_to_score = [c for c in CANDIDATES if needle in c.name.lower()]
        if not candidates_to_score:
            print(f"[!] No candidates match '{args.only}'. Available:")
            for c in CANDIDATES:
                print(f'    {c.name}')
            sys.exit(1)

    print(f'Scoring {len(candidates_to_score)} candidate(s) with '
          f'sample_size={args.sample_size}')
    print(f'Visual descriptor threshold: {VISUAL_DESCRIPTOR_THRESHOLD} '
          f'distinct visual words per review')

    cards = []
    t_start = time.time()
    for candidate in candidates_to_score:
        card = score_product(candidate, args.sample_size)
        cards.append(card)

    # Print detailed scorecards first
    for card in cards:
        print_scorecard(card)

    # Then the summary table
    print_summary_table(cards)

    elapsed = time.time() - t_start
    print(f'\n[OK] Completed in {elapsed:.0f}s total.')


if __name__ == '__main__':
    main()
