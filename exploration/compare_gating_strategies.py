'''compare_gating_strategies.py
----------------------------
Empirical comparison of two candidate 'visual-content gates' for the review
preprocessing pipeline:

  1. VOCAB GATE   — regex-match against the shared `visual_vocabulary.py`;
                    pass if >= VISUAL_WORD_THRESHOLD (=2) distinct words match.
  2. LLM GATE     — GPT-4o-mini binary judgment with brief reason (JSON out).

Runs both gates over a stratified 200-review sample from the chess_set pull
(60/20/20 across rating bands 5 / 4 / 1-3-pooled), writes a per-review CSV,
and prints a console summary with the confusion matrix plus disagreement
examples. Makes NO recommendation — the user reads the output and decides.

Usage:
    python compare_gating_strategies.py --dry-run   # prints prompt + cost estimate, no API calls
    python compare_gating_strategies.py             # real run

Outputs:
    data/chess_set/gate_comparison.csv
'''

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter

from dotenv import load_dotenv

from visual_vocabulary import passes_vocab_gate, VISUAL_WORD_THRESHOLD


## CONFIG ##

CHESS_REVIEWS_PATH = 'data/chess_set/reviews.jsonl'
OUTPUT_CSV_PATH = 'data/chess_set/gate_comparison.csv'

# Stratified sample: 60/20/20 across rating bands = 200 reviews total.
STRATIFICATION = [
    ('5-star', {5}, 120),
    ('4-star', {4}, 40),
    ('1-3-star', {1, 2, 3}, 40),
]
SEED = 20260417  # today's date, for reproducibility

LLM_MODEL = 'gpt-4o-mini'
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 200
LLM_RETRIES = 2  # retries on failure (on top of 1 initial attempt)

# gpt-4o-mini pricing as of 2024-2025 (USD per 1M tokens)
LLM_INPUT_PRICE = 0.15 / 1_000_000
LLM_OUTPUT_PRICE = 0.60 / 1_000_000
COST_ABORT_THRESHOLD = 2.00

LLM_SYSTEM_PROMPT = '''You are filtering Amazon product reviews to find ones with concrete visual content for an image-generation task.

PASS if the review describes how the product looks: colors, materials, shape, size, textures, finishes, decorative details, or visible parts. Complaints about appearance (e.g., "darker than pictured", "smaller than the photo") count as PASS.

FAIL if the review only covers non-visual topics: shipping/packaging, recipient reactions, durability/function without visual description, generic praise ("great product"), or price.

Respond with JSON only, no prose:
{"passes_gate": true or false, "reason": "one short sentence, 15 words max"}'''


## DATA LOADING + SAMPLING ##

def load_reviews(path: str) -> list[dict]:
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


def review_text(r: dict) -> str:
    '''Combine title + text into one string for both gates.'''
    title = (r.get('title') or '').strip()
    body = (r.get('text') or '').strip()
    return (title + ' ' + body).strip()


def synth_id(r: dict) -> str:
    '''Reviews have no native id; synthesize from user_id + timestamp.'''
    return f'{r.get("user_id", "unk")}_{r.get("timestamp", 0)}'


def stratified_sample(reviews: list[dict], seed: int) -> list[dict]:
    '''Return a list of reviews, stratified across rating bands per STRATIFICATION.'''
    by_band = {name: [] for name, _, _ in STRATIFICATION}
    for r in reviews:
        rating = r.get('rating')
        if rating is None:
            continue
        rating = int(rating)
        for name, ratings, _ in STRATIFICATION:
            if rating in ratings:
                by_band[name].append(r)
                break

    rng = random.Random(seed)
    sample = []
    for name, _, n in STRATIFICATION:
        pool = by_band[name]
        if len(pool) < n:
            print(f'  [!] Band {name}: pool={len(pool)}, requested={n}. Taking all.')
            sample.extend(pool)
        else:
            sample.extend(rng.sample(pool, n))

    return sample


## LLM GATE (with JSON repair, per HW02 lesson) ##

_JSON_BLOCK_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)


def repair_json(raw: str) -> dict:
    '''Try strict parse first; fall back to extracting the first {...} block.'''
    raw = raw.strip()
    # Strip common markdown fences.
    if raw.startswith('```'):
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f'Could not extract JSON from model output: {raw[:200]!r}')


def call_llm(client, text: str) -> tuple[dict, dict]:
    '''Returns (result, usage) where:
      result = {"passes_gate": bool|None, "reason": str}
      usage  = {"input_tokens": int, "output_tokens": int}
    On total failure, passes_gate is None and reason records the error.'''
    user_prompt = f'Review:\n"""\n{text}\n"""'
    last_err = None
    usage = {'input_tokens': 0, 'output_tokens': 0}

    for attempt in range(LLM_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                response_format={'type': 'json_object'},
                messages=[
                    {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_prompt},
                ],
            )
            if resp.usage:
                usage['input_tokens'] = resp.usage.prompt_tokens
                usage['output_tokens'] = resp.usage.completion_tokens
            raw = resp.choices[0].message.content or ''
            parsed = repair_json(raw)
            if 'passes_gate' not in parsed:
                raise ValueError(f'missing passes_gate key in {parsed!r}')
            return (
                {
                    'passes_gate': bool(parsed['passes_gate']),
                    'reason': str(parsed.get('reason', ''))[:300],
                },
                usage,
            )
        except Exception as e:
            last_err = e
            if attempt < LLM_RETRIES:
                time.sleep(1.0 * (attempt + 1))

    return (
        {'passes_gate': None, 'reason': f'ERROR: {type(last_err).__name__}: {last_err}'},
        usage,
    )


## COST ESTIMATION ##

def estimate_cost(sample_texts: list[str]) -> tuple[float, float, float]:
    '''Rough pre-flight estimate (chars/4 heuristic).
    Returns (est_input_tokens, est_output_tokens, est_usd).'''
    sys_tokens = len(LLM_SYSTEM_PROMPT) / 4
    wrapper_tokens = 20  # per-call framing around the review text
    est_in = sum(sys_tokens + wrapper_tokens + len(t) / 4 for t in sample_texts)
    est_out = len(sample_texts) * 50
    est_usd = est_in * LLM_INPUT_PRICE + est_out * LLM_OUTPUT_PRICE
    return est_in, est_out, est_usd


## OUTPUT: CSV + SUMMARY ##

CSV_FIELDS = [
    'review_id', 'rating', 'length',
    'vocab_decision', 'vocab_word_count', 'vocab_matches',
    'llm_decision', 'llm_reason',
    'agreement',
]


def write_csv(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def print_summary(rows: list[dict], sample_texts_by_id: dict):
    n = len(rows)
    # Drop rows where LLM errored for the 2x2 / agreement math.
    clean = [r for r in rows if r['llm_decision'] in ('pass', 'fail')]
    n_errors = n - len(clean)

    print('\n' + '=' * 78)
    print('  GATE COMPARISON SUMMARY')
    print('=' * 78)
    print(f'  Sample size:            {n}')
    if n_errors:
        print(f'  LLM errored on:         {n_errors}  (excluded from matrix)')

    vocab_pass = sum(1 for r in clean if r['vocab_decision'] == 'pass')
    llm_pass = sum(1 for r in clean if r['llm_decision'] == 'pass')
    print(f'  Vocab PASS:             {vocab_pass}/{len(clean)}  ({vocab_pass/len(clean)*100:.1f}%)')
    print(f'  LLM PASS:               {llm_pass}/{len(clean)}  ({llm_pass/len(clean)*100:.1f}%)')

    agree = sum(1 for r in clean if r['agreement'] == 'True')
    print(f'  Overall agreement:      {agree}/{len(clean)}  ({agree/len(clean)*100:.1f}%)')

    # 2x2 confusion
    vp_lp = sum(1 for r in clean if r['vocab_decision'] == 'pass' and r['llm_decision'] == 'pass')
    vp_lf = sum(1 for r in clean if r['vocab_decision'] == 'pass' and r['llm_decision'] == 'fail')
    vf_lp = sum(1 for r in clean if r['vocab_decision'] == 'fail' and r['llm_decision'] == 'pass')
    vf_lf = sum(1 for r in clean if r['vocab_decision'] == 'fail' and r['llm_decision'] == 'fail')

    print('\n  CONFUSION MATRIX (rows = vocab, cols = LLM):')
    print(f'                   LLM pass   LLM fail')
    print(f'    vocab pass      {vp_lp:>6}     {vp_lf:>6}')
    print(f'    vocab fail      {vf_lp:>6}     {vf_lf:>6}')

    # Per-band agreement
    print('\n  AGREEMENT BY RATING BAND:')
    bands = [('5-star', {'5'}), ('4-star', {'4'}), ('1-3-star', {'1', '2', '3'})]
    for name, rats in bands:
        sub = [r for r in clean if r['rating'] in rats]
        if not sub:
            continue
        a = sum(1 for r in sub if r['agreement'] == 'True')
        print(f'    {name:<10} {a}/{len(sub)}  ({a/len(sub)*100:.1f}%)')

    # Disagreement samples — stratified across rating bands so we can tell
    # whether disagreements cluster in a specific band.
    def stratified_picks(candidates: list, targets=(('5', 2), ('4', 2), ('1-3', 1))):
        '''Pick a band-diverse subset. Backfill shortfall from whichever band has extras.'''
        band_of = {'5': {'5'}, '4': {'4'}, '1-3': {'1', '2', '3'}}
        by_band = {name: [] for name, _ in targets}
        for r in candidates:
            for name, rats in band_of.items():
                if r['rating'] in rats:
                    by_band[name].append(r)
                    break
        picks = []
        shortfall = 0
        for name, n in targets:
            pool = by_band[name]
            take = min(n, len(pool))
            picks.extend(pool[:take])
            shortfall += n - take
        if shortfall:
            leftovers = []
            for name, n in targets:
                pool = by_band[name]
                if len(pool) > n:
                    leftovers.extend(pool[n:])
            picks.extend(leftovers[:shortfall])
        return picks

    def show(header: str, predicate):
        print(f'\n  {header}')
        cands = [r for r in clean if predicate(r)]
        picks = stratified_picks(cands)
        if not picks:
            print('    (none)')
            return
        print(f'    [total disagreements in this quadrant: {len(cands)}]')
        for i, r in enumerate(picks, 1):
            text = sample_texts_by_id.get(r['review_id'], '')
            snippet = text[:220].replace('\n', ' ')
            if len(text) > 220:
                snippet += '...'
            print(f'    [{i}] rating={r["rating"]}  len={r["length"]}')
            print(f'        vocab_matches: {r["vocab_matches"] or "(none)"}')
            print(f'        llm_reason:    {r["llm_reason"]}')
            print(f'        text: {snippet}')
            print()

    show('VOCAB=pass, LLM=fail  (possible vocab false positives):',
         lambda r: r['vocab_decision'] == 'pass' and r['llm_decision'] == 'fail')
    show('VOCAB=fail, LLM=pass  (vocab false negatives — costlier errors):',
         lambda r: r['vocab_decision'] == 'fail' and r['llm_decision'] == 'pass')


## MAIN ##

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print prompt + cost estimate + vocab previews, skip LLM calls.')
    args = parser.parse_args()

    if not os.path.exists(CHESS_REVIEWS_PATH):
        print(f'[!] {CHESS_REVIEWS_PATH} not found')
        sys.exit(1)

    print(f'Loading reviews from {CHESS_REVIEWS_PATH} ...')
    reviews = load_reviews(CHESS_REVIEWS_PATH)
    with_text = [r for r in reviews if (r.get('text') or '').strip()]
    print(f'  total={len(reviews)}  with_text={len(with_text)}')

    sample = stratified_sample(with_text, SEED)
    print(f'  sampled={len(sample)} (seed={SEED})')

    # Run vocab gate on the sample (cheap, deterministic).
    vocab_rows = []
    for r in sample:
        text = review_text(r)
        passes, matches = passes_vocab_gate(text)
        vocab_rows.append({
            'review': r,
            'text': text,
            'vocab_pass': passes,
            'vocab_matches': sorted(matches),
        })

    # Cost estimate
    texts = [v['text'] for v in vocab_rows]
    est_in, est_out, est_usd = estimate_cost(texts)
    print(f'\nEstimated LLM cost: ${est_usd:.4f}  (in~{est_in:.0f} tokens, out~{est_out:.0f})')

    # DRY RUN: print prompt + examples and exit.
    if args.dry_run:
        print('\n' + '=' * 78)
        print('  LLM SYSTEM PROMPT')
        print('=' * 78)
        print(LLM_SYSTEM_PROMPT)
        print('\n' + '=' * 78)
        print('  EXAMPLE USER PROMPT (first sampled review)')
        print('=' * 78)
        print(f'Review:\n"""\n{vocab_rows[0]["text"]}\n"""')

        n_pass = sum(1 for v in vocab_rows if v['vocab_pass'])
        print('\n' + '=' * 78)
        print('  VOCAB-GATE RESULTS (preview, LLM not yet called)')
        print('=' * 78)
        print(f'  pass: {n_pass}/{len(vocab_rows)}  ({n_pass/len(vocab_rows)*100:.1f}%)')

        print('\n  3 random PASS examples:')
        passed = [v for v in vocab_rows if v['vocab_pass']]
        for v in passed[:3]:
            snip = v['text'][:200].replace('\n', ' ')
            print(f'    matches={v["vocab_matches"][:10]}')
            print(f'    text: {snip}{"..." if len(v["text"]) > 200 else ""}\n')

        print('  3 random FAIL examples:')
        failed = [v for v in vocab_rows if not v['vocab_pass']]
        for v in failed[:3]:
            snip = v['text'][:200].replace('\n', ' ')
            print(f'    matches={v["vocab_matches"]}')
            print(f'    text: {snip}{"..." if len(v["text"]) > 200 else ""}\n')

        print('Dry run complete. Re-run without --dry-run to call the LLM.')
        return

    # Cost guardrail.
    if est_usd > COST_ABORT_THRESHOLD:
        print(f'[!] Estimated ${est_usd:.2f} exceeds ${COST_ABORT_THRESHOLD} budget. Aborting.')
        sys.exit(1)

    # REAL RUN: load key + client.
    load_dotenv()
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print('[!] OPENAI_API_KEY not set (checked env + .env).')
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print('[!] `openai` package not installed. Run: pip install openai python-dotenv')
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print('\nCalling GPT-4o-mini on 200 reviews...')
    rows = []
    sample_texts_by_id = {}
    total_in = total_out = 0
    n_errors = 0
    t0 = time.time()

    for i, v in enumerate(vocab_rows, 1):
        r = v['review']
        rid = synth_id(r)
        llm_result, usage = call_llm(client, v['text'])
        total_in += usage['input_tokens']
        total_out += usage['output_tokens']

        vd = 'pass' if v['vocab_pass'] else 'fail'
        if llm_result['passes_gate'] is None:
            ld = 'error'
            n_errors += 1
        else:
            ld = 'pass' if llm_result['passes_gate'] else 'fail'
        agreement = str(vd == ld) if ld != 'error' else ''

        rows.append({
            'review_id': rid,
            'rating': str(int(r.get('rating', 0))),
            'length': len(v['text']),
            'vocab_decision': vd,
            'vocab_word_count': len(v['vocab_matches']),
            'vocab_matches': ';'.join(v['vocab_matches']),
            'llm_decision': ld,
            'llm_reason': llm_result['reason'],
            'agreement': agreement,
        })
        sample_texts_by_id[rid] = v['text']

        if i % 10 == 0 or i == len(vocab_rows):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            print(f'  [{i:>3}/{len(vocab_rows)}] {rate:.1f} rev/s  errors={n_errors}')

    # Actual cost from reported usage
    actual_usd = total_in * LLM_INPUT_PRICE + total_out * LLM_OUTPUT_PRICE
    print(f'\nActual token usage: in={total_in}  out={total_out}  cost=${actual_usd:.4f}')

    write_csv(rows, OUTPUT_CSV_PATH)
    print(f'Wrote {len(rows)} rows to {OUTPUT_CSV_PATH}')

    print_summary(rows, sample_texts_by_id)


if __name__ == '__main__':
    main()
