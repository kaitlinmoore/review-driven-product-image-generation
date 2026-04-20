''' build_filter_cache.py
---------------------
Run the LLM visual-content gate on the full review set for each product and
cache the decisions to disk.

Inputs:
    data/{product}/reviews.jsonl

Outputs:
    data/filter_caches/{product}_filter_decisions_v1.json

The output JSON is keyed by a synthesized review id: f'{user_id}_{timestamp}'.
Every 50 new decisions are flushed atomically to disk (write .tmp, os.replace)
so a crash loses at most 49 calls. On re-run, reviews already in the cache are
skipped. Safe to stop and restart at any time.

Usage and CLI flags:
    python build_filter_cache.py --dry-run              # Show prompt and cost estimate, no API calls.
    python build_filter_cache.py                        # Run all three canonical products.
    python build_filter_cache.py --only chess           # Run one product.
    python build_filter_cache.py --workers 8            # e.g. 8 concurrent LLM calls (default 1)

Cost guardrail: Currently aborts if aggregate estimate exceeds $3 across selected products. Shouldn't come close.
'''

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time

# Allow local parallelized work.
from concurrent.futures import ThreadPoolExecutor, as_completed

# Use for local .env defined API keys.
from dotenv import load_dotenv


# ============================================================================
# v1 FILTER PROMPT | DO NOT EDIT WITHOUT BUMPING VERSION
#
# Cached decisions keyed by prompt_version='v1' are only valid against this exact
# prompt text. If you want to try a different prompt, bump to v2 and write to
# filter_decisions_v2.json so existing decisions aren't invalidated. This is
# critical for replay.
# ============================================================================
PROMPT_VERSION = 'v1'

LLM_SYSTEM_PROMPT_V1 = '''You are filtering Amazon product reviews to find ones with concrete visual content for an image-generation task.

PASS if the review describes how the product looks: colors, materials, shape, size, textures, finishes, decorative details, or visible parts. Complaints about appearance (e.g., "darker than pictured", "smaller than the photo") count as PASS.

FAIL if the review only covers non-visual topics: shipping/packaging, recipient reactions, durability/function without visual description, generic praise ("great product"), or price.

Respond with JSON only, no prose:
{"passes_gate": true or false, "reason": "one short sentence, 15 words max"}'''

# SHA-256 of the prompt text above. Cached decisions carry this hash in the
# header. If the prompt is edited without bumping PROMPT_VERSION, the hash
# will mismatch on next load and the run will refuse to merge. This prevents
# silent cross-prompt contamination under the same version label. Learned this
# lesson the hard way in the past.
PROMPT_SHA256 = hashlib.sha256(LLM_SYSTEM_PROMPT_V1.encode('utf-8')).hexdigest()


## CONFIG ##

DATA_DIR = 'data'
FILTER_CACHE_DIR = 'data/filter_caches'


def cache_path_for(slug: str) -> str:
    '''Canonical path for a product's filter-decision cache.'''
    return os.path.join(FILTER_CACHE_DIR, f'{slug}_filter_decisions_{PROMPT_VERSION}.json')


LLM_MODEL = 'gpt-4o-mini'
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 200
LLM_RETRIES = 2  # on top of the initial attempt

# TODO: Confirm this is current.
LLM_INPUT_PRICE = 0.15 / 1_000_000
LLM_OUTPUT_PRICE = 0.60 / 1_000_000
COST_ABORT_THRESHOLD = 3.00

FLUSH_EVERY = 50
REVIEW_ID_SCHEME = 'user_id_timestamp'


## DISCOVERY ##

def discover_products() -> dict[str, str]:
    '''Return {slug: dir} for each data/*/ that has summary.json and a
    non-empty reviews.jsonl. The summary.json presence signals a completed
    pull, so partial pulls are skipped to avoid wasting LLM calls.'''
    out: dict[str, str] = {}
    if not os.path.isdir(DATA_DIR):
        return out
    for name in sorted(os.listdir(DATA_DIR)):
        pdir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(pdir):
            continue
        reviews = os.path.join(pdir, 'reviews.jsonl')
        summary = os.path.join(pdir, 'summary.json')
        if not os.path.exists(reviews) or os.path.getsize(reviews) == 0:
            continue
        if not os.path.exists(summary):
            continue  # pull in progress
        out[name] = pdir
    return out


def resolve_only(products: dict[str, str], needle: str) -> dict[str, str]:
    '''Substring-match --only against discovered slugs. Require exactly 1 match.'''
    matches = [k for k in products if needle.lower() in k.lower()]
    if len(matches) == 0:
        print(f'[!] --only {needle!r} matches no product. '
              f'Available: {sorted(products)}')
        sys.exit(1)
    if len(matches) > 1:
        print(f'[!] --only {needle!r} matches multiple: {matches}. '
              f'Be more specific.')
        sys.exit(1)
    return {matches[0]: products[matches[0]]}


## REVIEW LOADING and ID SYNTHESIS ##

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
    title = (r.get('title') or '').strip()
    body = (r.get('text') or '').strip()
    return (title + ' ' + body).strip()


def synth_id(r: dict) -> str:
    return f"{r.get('user_id', 'unk')}_{r.get('timestamp', 0)}"


## LLM CALL and JSON REPAIR ##

_JSON_BLOCK_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)


def repair_json(raw: str) -> dict:
    raw = raw.strip()
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
    raise ValueError(f'Could not extract JSON: {raw[:200]!r}')


def call_llm(client, text: str) -> tuple[dict, dict]:
    '''Returns ({passes_gate, reason}, {input_tokens, output_tokens}).'''
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
                    {'role': 'system', 'content': LLM_SYSTEM_PROMPT_V1},
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
        {'passes_gate': None, 'reason': f'LLM error: {type(last_err).__name__}: {last_err}'},
        usage,
    )


## CACHE STATE: thread-safe, atomic incremental saves ##

class FilterCache:
    def __init__(self, path: str, total_reviews: int, dropped_empty_text: int):
        self.path = path
        self.total_reviews = total_reviews
        self.dropped_empty_text = dropped_empty_text
        self.decisions: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.new_since_flush = 0
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Version and hash check. Bug I learned the hard way: just checking
            # prompt_version string isn't enough. Someone can edit the prompt
            # text without bumping version and silently mix decisions.
            existing_version = data.get('prompt_version')
            existing_sha = data.get('prompt_sha256')
            if existing_version and existing_version != PROMPT_VERSION:
                print(f'  [!] {self.path} was built with prompt_version='
                      f'{existing_version!r}, expected {PROMPT_VERSION!r}. '
                      f'Move or delete it before re-running.')
                sys.exit(1)
            if existing_version and existing_sha != PROMPT_SHA256:
                print(f'  [!] {self.path} prompt_sha256 mismatch '
                      f'(cached={existing_sha!r}, current={PROMPT_SHA256[:16]}...). '
                      f'Bump PROMPT_VERSION and write to a new file (v2) instead of merging under v1.')
                sys.exit(1)
            self.decisions = data.get('decisions', {})
            n_err = sum(1 for d in self.decisions.values() if d.get('passes_gate') is None)
            print(f'  Resuming from cache: {len(self.decisions):,} decisions loaded'
                  f"{f' ({n_err} errored — will be retried)' if n_err else ''}")
        except (json.JSONDecodeError, OSError) as e:
            print(f'  [!] Existing cache unreadable ({e}); starting fresh.')
            self.decisions = {}

    def has(self, rid: str) -> bool:
        '''True only for a non-error cached decision. Errored entries
        (passes_gate=None) are intentionally re-tried on resume so a rate-limit
        or transient failure doesn't leave permanent holes in the cache.'''
        with self.lock:
            d = self.decisions.get(rid)
            return d is not None and d.get('passes_gate') is not None

    def record(self, rid: str, passes: bool | None, reason: str):
        with self.lock:
            self.decisions[rid] = {'passes_gate': passes, 'reason': reason}
            self.new_since_flush += 1
            if self.new_since_flush >= FLUSH_EVERY:
                self._flush_locked()
                self.new_since_flush = 0

    def _payload(self) -> dict:
        passes = sum(1 for d in self.decisions.values() if d.get('passes_gate') is True)
        fails = sum(1 for d in self.decisions.values() if d.get('passes_gate') is False)
        errors = sum(1 for d in self.decisions.values() if d.get('passes_gate') is None)
        return {
            'prompt_version': PROMPT_VERSION,
            'prompt_sha256': PROMPT_SHA256,
            'model': LLM_MODEL,
            'temperature': LLM_TEMPERATURE,
            'review_id_scheme': REVIEW_ID_SCHEME,
            'total_reviews': self.total_reviews,
            'dropped_empty_text': self.dropped_empty_text,
            'passes_gate': passes,
            'fails_gate': fails,
            'errors': errors,
            'decisions': self.decisions,
        }

    def _flush_locked(self):
        '''Caller must hold self.lock.'''
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self._payload(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def flush(self):
        with self.lock:
            self._flush_locked()
            self.new_since_flush = 0

    def counts(self) -> tuple[int, int, int]:
        with self.lock:
            passes = sum(1 for d in self.decisions.values() if d.get('passes_gate') is True)
            fails = sum(1 for d in self.decisions.values() if d.get('passes_gate') is False)
            errors = sum(1 for d in self.decisions.values() if d.get('passes_gate') is None)
        return passes, fails, errors


## COST / TIME ESTIMATION ##

def estimate_one_product(reviews_with_text: list[dict]) -> tuple[float, float, float]:
    '''Returns (input_tokens, output_tokens, usd).'''
    sys_tokens = len(LLM_SYSTEM_PROMPT_V1) / 4
    wrapper_tokens = 20
    in_toks = sum(sys_tokens + wrapper_tokens + len(review_text(r)) / 4 for r in reviews_with_text)
    out_toks = len(reviews_with_text) * 50
    usd = in_toks * LLM_INPUT_PRICE + out_toks * LLM_OUTPUT_PRICE
    return in_toks, out_toks, usd


## PER-PRODUCT DRIVER ##

def run_product(product_key: str, product_dir: str, args, client) -> dict:
    reviews_path = os.path.join(product_dir, 'reviews.jsonl')
    cache_path = cache_path_for(product_key)
    os.makedirs(FILTER_CACHE_DIR, exist_ok=True)

    if not os.path.exists(reviews_path):
        print(f'  [!] {reviews_path} not found; skipping')
        return {'product': product_key, 'skipped': True}

    print(f'\n[{product_key}] loading {reviews_path}')
    all_reviews = load_reviews(reviews_path)
    with_text = [r for r in all_reviews if (r.get('text') or '').strip()]
    dropped = len(all_reviews) - len(with_text)
    print(f'  total={len(all_reviews):,}  with_text={len(with_text):,}  '
          f'dropped_empty_text={dropped}')

    cache = FilterCache(cache_path, total_reviews=len(with_text), dropped_empty_text=dropped)

    to_process = [r for r in with_text if not cache.has(synth_id(r))]
    already = len(with_text) - len(to_process)
    if already:
        print(f'  skipping {already:,} reviews already cached')
    print(f'  will call LLM on {len(to_process):,} reviews')

    if args.dry_run:
        return {
            'product': product_key,
            'dry_run': True,
            'to_process': len(to_process),
            'already_cached': already,
        }

    if not to_process:
        cache.flush()  # ensure file reflects current state even if nothing to do
        passes, fails, errors = cache.counts()
        print(f'  nothing to do. passes={passes} fails={fails} errors={errors}')
        return {
            'product': product_key, 'passes': passes, 'fails': fails,
            'errors': errors, 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0.0,
        }

    # Process
    total_in = 0
    total_out = 0
    n_done = 0
    n_err = 0
    token_lock = threading.Lock()
    t0 = time.time()

    def worker(review: dict) -> tuple[bool, int, int]:
        rid = synth_id(review)
        text = review_text(review)
        result, usage = call_llm(client, text)
        cache.record(rid, result['passes_gate'], result['reason'])
        is_err = result['passes_gate'] is None
        return is_err, usage['input_tokens'], usage['output_tokens']

    if args.workers <= 1:
        for i, r in enumerate(to_process, 1):
            is_err, in_tok, out_tok = worker(r)
            with token_lock:
                total_in += in_tok
                total_out += out_tok
                if is_err:
                    n_err += 1
            n_done = i
            if i % 50 == 0 or i == len(to_process):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(to_process) - i) / max(rate, 1e-6)
                print(f'    [{i:>5}/{len(to_process):,}] {rate:.1f} rev/s  '
                      f'errors={n_err}  eta={eta/60:.1f} min')
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(worker, r) for r in to_process]
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    is_err, in_tok, out_tok = fut.result()
                except Exception as e:
                    # Shouldn't reach here — worker catches inside call_llm.
                    print(f'    [!] worker raised: {e}')
                    continue
                with token_lock:
                    total_in += in_tok
                    total_out += out_tok
                    if is_err:
                        n_err += 1
                n_done = i
                if i % 50 == 0 or i == len(to_process):
                    elapsed = time.time() - t0
                    rate = i / max(elapsed, 1e-6)
                    eta = (len(to_process) - i) / max(rate, 1e-6)
                    print(f'    [{i:>5}/{len(to_process):,}] {rate:.1f} rev/s  '
                          f'errors={n_err}  eta={eta/60:.1f} min')

    cache.flush()
    passes, fails, errors = cache.counts()
    cost = total_in * LLM_INPUT_PRICE + total_out * LLM_OUTPUT_PRICE
    print(f'  [{product_key}] done: passes={passes:,} fails={fails:,} errors={errors} '
          f'cost=${cost:.4f}')

    return {
        'product': product_key,
        'passes': passes,
        'fails': fails,
        'errors': errors,
        'input_tokens': total_in,
        'output_tokens': total_out,
        'cost_usd': cost,
        'processed_this_run': n_done,
    }


## MAIN ##

def main():
    parser = argparse.ArgumentParser(description='Build the v1 LLM filter-decision cache.')
    parser.add_argument('--only', default=None,
                        help="Substring-match a single product slug (e.g. 'chess' or 'backpack'). "
                             'Default: run all discovered products.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Concurrent LLM calls (default 1; try 8 for wall-clock speedup).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show prompt + cost/time estimate and exit.')
    args = parser.parse_args()

    products = discover_products()
    if not products:
        print(f'[!] No products found under {DATA_DIR}/ '
              f'(need reviews.jsonl + summary.json).')
        sys.exit(1)
    print(f'Discovered products: {sorted(products)}')

    selected = resolve_only(products, args.only) if args.only else products

    # Load reviews for estimation up-front.
    per_product_reviews = {}
    total_tokens_in = 0.0
    total_tokens_out = 0.0
    total_usd = 0.0
    total_to_process = 0

    for key, pdir in selected.items():
        path = os.path.join(pdir, 'reviews.jsonl')
        if not os.path.exists(path):
            print(f'  [!] {path} not found; will skip at runtime')
            per_product_reviews[key] = None
            continue
        raw = load_reviews(path)
        with_text = [r for r in raw if (r.get('text') or '').strip()]
        per_product_reviews[key] = with_text

        # For resumable estimates, only count reviews we haven't SUCCESSFULLY
        # decided yet. Errored entries (passes_gate=None) count as todo.
        cache_path = cache_path_for(key)
        done_ids: set[str] = set()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f).get('decisions', {})
                done_ids = {k for k, v in existing.items()
                            if v.get('passes_gate') is not None}
            except (json.JSONDecodeError, OSError):
                pass

        todo = [r for r in with_text if synth_id(r) not in done_ids]
        in_toks, out_toks, usd = estimate_one_product(todo)
        total_tokens_in += in_toks
        total_tokens_out += out_toks
        total_usd += usd
        total_to_process += len(todo)

        print(f'  {key:<6} total_with_text={len(with_text):,}  '
              f'already_cached={len(with_text) - len(todo):,}  '
              f'to_process={len(todo):,}  est_cost=${usd:.4f}')

    # Time estimates
    per_call_seconds = 0.67  # empirical from the comparison run
    seq_seconds = total_to_process * per_call_seconds
    parallel_seconds = seq_seconds / max(args.workers, 1)

    print(f'\n  aggregate: to_process={total_to_process:,}  '
          f'est_tokens in~{total_tokens_in:.0f} out~{total_tokens_out:.0f}  '
          f'est_cost=${total_usd:.4f}')
    print(f'  time estimate: sequential ~{seq_seconds/60:.1f} min  '
          f'| with --workers={args.workers} ~{parallel_seconds/60:.1f} min')

    if args.dry_run:
        print('\n' + '=' * 78)
        print('  v1 FILTER PROMPT (system role)')
        print('=' * 78)
        print(LLM_SYSTEM_PROMPT_V1)
        print('\n' + '=' * 78)
        print('  EXAMPLE USER PROMPT (first review that would be processed)')
        print('=' * 78)
        example = None
        for key, revs in per_product_reviews.items():
            if not revs:
                continue
            cache_path = cache_path_for(key)
            done_ids: set[str] = set()
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f).get('decisions', {})
                    done_ids = {k for k, v in existing.items()
                                if v.get('passes_gate') is not None}
                except (json.JSONDecodeError, OSError):
                    pass
            for r in revs:
                if synth_id(r) not in done_ids:
                    example = r
                    break
            if example:
                break
        if example:
            print(f'Review:\n"""\n{review_text(example)}\n"""')
        else:
            print('  (no uncached reviews to show)')

        print('\n' + '=' * 78)
        print('  CACHE FILE STRUCTURE (what will be written)')
        print('=' * 78)
        sample = {
            'prompt_version': PROMPT_VERSION,
            'model': LLM_MODEL,
            'temperature': LLM_TEMPERATURE,
            'review_id_scheme': REVIEW_ID_SCHEME,
            'total_reviews': '<int>',
            'dropped_empty_text': '<int>',
            'passes_gate': '<int>',
            'fails_gate': '<int>',
            'errors': '<int>',
            'decisions': {
                '<user_id>_<timestamp>': {'passes_gate': True, 'reason': '...'},
                '<user_id>_<timestamp>': {'passes_gate': False, 'reason': '...'},
            },
        }
        print(json.dumps(sample, indent=2))
        print(f'\n  Incremental save every {FLUSH_EVERY} decisions (atomic: .tmp + os.replace).')
        print(f'  Resumable: existing {FILTER_CACHE_DIR}/{{slug}}_filter_decisions_{PROMPT_VERSION}.json is merged into, never overwritten.')
        print('\nDry-run complete. Re-run without --dry-run to call the LLM.')
        return

    # Cost guardrail
    if total_usd > COST_ABORT_THRESHOLD:
        print(f'[!] Estimated ${total_usd:.2f} exceeds ${COST_ABORT_THRESHOLD} budget. Aborting.')
        sys.exit(1)

    # Real run: load key and client
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

    t_all = time.time()
    summaries = []
    for key, pdir in selected.items():
        try:
            summary = run_product(key, pdir, args, client)
            summaries.append(summary)
        except KeyboardInterrupt:
            print('\n[!] Interrupted. Cache has been flushed at last checkpoint; re-run to resume.')
            sys.exit(130)

    elapsed = time.time() - t_all
    print('\n' + '=' * 78)
    print('  ALL PRODUCTS DONE')
    print('=' * 78)
    grand_cost = 0.0
    for s in summaries:
        if s.get('skipped') or s.get('dry_run'):
            continue
        print(f"  {s['product']:<6} passes={s['passes']:>5,}  fails={s['fails']:>5,}  "
              f"errors={s['errors']:>3}  cost=${s['cost_usd']:.4f}")
        grand_cost += s['cost_usd']
    print(f'  total cost: ${grand_cost:.4f}  wall-clock: {elapsed/60:.1f} min')


if __name__ == '__main__':
    main()
