'''
VLM-as-judge spike test.

Not wired into the pipeline. Exists to answer one question before
committing to a full v6 experiment:

  Does gpt-4o vision produce calibrated, discriminating scores when
  asked to rate similarity between a ground-truth product image and
  a generated candidate?

If yes -- VLM-as-judge is viable as an in-loop quality signal for a
future experimental config. If no (scores anchor to 0.5 / 1.0 /
random, or don't discriminate between known-better and known-worse
image pairs), save the time; don't run v6.

Uses existing v5 iteration images from committed data/. No new
image generation. Cost: ~$0.04 per full spike run (cached after
first pass).

Usage:
    python exploration/vlm_judge_spike.py
    # or, with a named subset:
    python exploration/vlm_judge_spike.py --only baseline,jeans_gpt

Reads OPENAI_API_KEY from .env in the repo root.
'''

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from dotenv import load_dotenv
from openai import OpenAI

from replay import cached_call

try:
    load_dotenv(ROOT / '.env')
except Exception:
    pass

## CONSTANTS ##

JUDGE_MODEL = 'gpt-4o'
JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS = 400

JUDGE_SYSTEM = (
    'You are a careful product-image evaluator. Your job is to judge how '
    'faithfully a generated image depicts a target product, regardless of '
    'how the product is staged or presented. Ground your score in '
    'product-level visual evidence; use the full 0.00-1.00 range as the '
    'evidence warrants. Do not anchor to 0.5 or 1.0.'
)

JUDGE_PROMPT = '''You will see two images of a product.
Image A: the actual product (reference photograph).
Image B: a generated image attempting to depict the same product.

Your job is to rate how faithfully Image B depicts the product itself,
regardless of how it is staged or presented.

Focus ONLY on the product's intrinsic visual properties:
  - Color, wash, finish, material appearance
  - Shape, silhouette, proportions
  - Texture and construction details (stitching, hardware, patterns)
  - Logos, branding, text on the product itself
  - Distinctive features

IGNORE these (they are NOT product-level differences):
  - Composition and framing (retail layout vs lifestyle vs studio)
  - Background, props, lighting, camera angle
  - Whether a model is present
  - Image cropping, aspect ratio
  - Text overlays, watermarks, marketing elements

Step 1: List 3-5 product-level differences between B's and A's
depiction of the product. If the products are depicted equivalently
despite different framing, the list may be very short.

Step 2: Rate how faithfully Image B depicts the same product as
Image A, as an INTEGER from 0 to 100:
  0   = wrong product entirely
  100 = visually identical product depiction

Your score should reflect how much of the product's identity
(color, shape, material, details, branding) Image B preserves.

Use the full 0-100 range. Different image pairs should produce
distinctly different scores -- do NOT cluster at round numbers
like 50, 65, 70, 75. Scores like 37, 52, 63, 81 are preferred
over 25, 50, 75 when the evidence supports them.

Return the score as an integer.

JSON only:
{
  "differences": ["diff1", "diff2", "..."],
  "score": <integer 0-100>
}
'''


## HELPERS ##

def image_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = 'image/jpeg'
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'data:{mime};base64,{b64}'


def vlm_judge(ground_truth_path: str, generated_path: str) -> dict:
    '''Compare generated image to ground truth. Returns parsed JSON
    with `differences` (list of strings) and `score` (float in [0, 1]).

    Cached via replay_cache so repeated invocations are free.
    '''
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f'ground truth image missing: {ground_truth_path}')
    if not os.path.exists(generated_path):
        raise FileNotFoundError(f'generated image missing: {generated_path}')

    # Cache key inputs: hashes of both images + prompt text + model + temperature.
    with open(ground_truth_path, 'rb') as f:
        gt_bytes = f.read()
    with open(generated_path, 'rb') as f:
        gen_bytes = f.read()

    inputs = {
        'kind': 'vlm_judge',
        'model': JUDGE_MODEL,
        'temperature': JUDGE_TEMPERATURE,
        'system_prompt': JUDGE_SYSTEM,
        'user_prompt': JUDGE_PROMPT,
        'ground_truth_sha256': _sha256(gt_bytes),
        'generated_sha256': _sha256(gen_bytes),
    }

    def _live() -> dict:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': JUDGE_SYSTEM},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': JUDGE_PROMPT},
                        {'type': 'image_url',
                         'image_url': {'url': image_data_url(ground_truth_path)}},
                        {'type': 'image_url',
                         'image_url': {'url': image_data_url(generated_path)}},
                    ],
                },
            ],
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        # Score is now an integer 0-100; divide by 100 for [0, 1].
        raw_score = parsed.get('score', 0)
        try:
            score = float(raw_score) / 100.0
        except (TypeError, ValueError):
            score = 0.0
        # Clamp defensively in case the VLM returns something out of range.
        score = max(0.0, min(1.0, score))
        return {
            'differences': parsed.get('differences', []),
            'score': score,
            'score_raw_integer': raw_score,
            'raw': content,
        }

    return cached_call('vlm_judge', inputs, _live, format='json')


def _sha256(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


## SPIKE TEST CASES ##

def path_gt(slug: str) -> str:
    return str(ROOT / 'data' / slug / 'images' / 'main.jpg')

def path_iter(slug: str, model: str, iteration: int) -> str:
    return str(
        ROOT / 'data' / slug / 'trajectories'
        / f'{model}_v5_unreachable' / f'iteration_{iteration}'
        / 'generated_image.png'
    )


SPIKE_CASES = [
    # Identical-image baseline -- should score ~1.0 if the metric has
    # any notion of perfect match.
    {
        'name': 'baseline_identical',
        'description': 'Same image for both slots (chess_set main.jpg twice). '
                       'Expect score ~= 1.0.',
        'gt_path': path_gt('chess_set'),
        'gen_path': path_gt('chess_set'),
        'expectation': 'very high (>= 0.90)',
    },

    # Cross-product negative -- should score very low.
    {
        'name': 'cross_product_negative',
        'description': 'chess_set main.jpg vs. a generated jeans image. '
                       'Different product category entirely. Expect score very low.',
        'gt_path': path_gt('chess_set'),
        'gen_path': path_iter('jeans', 'gpt', 0),
        'expectation': 'very low (< 0.20)',
    },

    # Known-positive from v5: jeans gpt Family B CLIP delta= = +0.22 across
    # iter_0 → iter_4. The VLM judge should reflect this improvement.
    {
        'name': 'jeans_gpt_iter0',
        'description': 'jeans main.jpg vs. v5 jeans/gpt iteration_0 (pre-refinement).',
        'gt_path': path_gt('jeans'),
        'gen_path': path_iter('jeans', 'gpt', 0),
        'expectation': 'baseline for jeans_gpt_iter4 comparison',
    },
    {
        'name': 'jeans_gpt_iter4',
        'description': 'jeans main.jpg vs. v5 jeans/gpt iteration_4 (after 4 retunes). '
                       'Family B said this is closer to GT than iter_0 (CLIP +0.22). '
                       'VLM judge should agree: iter_4 score > iter_0 score.',
        'gt_path': path_gt('jeans'),
        'gen_path': path_iter('jeans', 'gpt', 4),
        'expectation': 'strictly > jeans_gpt_iter0 score',
    },

    # Known-negative from v5: headphones flux all three Family B delta=
    # negative across iter_0 → iter_4. VLM judge should reflect decline.
    {
        'name': 'headphones_flux_iter0',
        'description': 'headphones main.jpg vs. v5 headphones/flux iteration_0.',
        'gt_path': path_gt('headphones'),
        'gen_path': path_iter('headphones', 'flux', 0),
        'expectation': 'baseline for headphones_flux_iter4 comparison',
    },
    {
        'name': 'headphones_flux_iter4',
        'description': 'headphones main.jpg vs. v5 headphones/flux iteration_4. '
                       'Family B said iter_4 is worse than iter_0 on all 3 encoders. '
                       'VLM judge should agree: iter_4 score <= iter_0 score.',
        'gt_path': path_gt('headphones'),
        'gen_path': path_iter('headphones', 'flux', 4),
        'expectation': 'strictly < headphones_flux_iter0 score',
    },

    # Known-flat from v5: chess_set flux controller truncated
    # (iters_taken=[1,1,0,0,0]). iter_0 and iter_4 should be visually
    # near-identical (iterations 2-4 got no inner refinement).
    {
        'name': 'chess_set_flux_iter0',
        'description': 'chess_set main.jpg vs. v5 chess_set/flux iteration_0.',
        'gt_path': path_gt('chess_set'),
        'gen_path': path_iter('chess_set', 'flux', 0),
        'expectation': 'baseline for chess_set_flux_iter4 comparison',
    },
    {
        'name': 'chess_set_flux_iter4',
        'description': 'chess_set main.jpg vs. v5 chess_set/flux iteration_4. '
                       'Controller truncated after image 1; iter_4 is a stochastic '
                       'regeneration without refinement. Expect ~= iter_0 score.',
        'gt_path': path_gt('chess_set'),
        'gen_path': path_iter('chess_set', 'flux', 4),
        'expectation': 'approximately equal to chess_set_flux_iter0 score',
    },
]


## RUNNER ##

def run_spike(only: list = None) -> list:
    results = []
    for case in SPIKE_CASES:
        if only and case['name'] not in only:
            continue
        print(f"\n[{case['name']}]")
        print(f"  {case['description']}")
        print(f"  Expectation: {case['expectation']}")
        try:
            out = vlm_judge(case['gt_path'], case['gen_path'])
            print(f"  SCORE: {out['score']:.3f}")
            print(f"  Differences cited:")
            for d in out['differences']:
                print(f"    - {d}")
            results.append({
                'name': case['name'],
                'score': out['score'],
                'differences': out['differences'],
                'expectation': case['expectation'],
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'name': case['name'],
                'score': None,
                'error': str(e),
            })
    return results


def print_summary(results: list) -> None:
    print('\n' + '=' * 72)
    print('SPIKE SUMMARY')
    print('=' * 72)
    print(f'{"case":<30} {"score":>8}  expectation')
    print('-' * 72)
    for r in results:
        score_str = f'{r["score"]:.3f}' if r.get('score') is not None else 'ERROR'
        print(f'{r["name"]:<30} {score_str:>8}  {r.get("expectation", "")}')

    # Interpretation checklist
    print('\n' + '-' * 72)
    print('VIABILITY CHECK')
    print('-' * 72)
    by_name = {r['name']: r for r in results if r.get('score') is not None}

    # 1. Identical-image baseline
    b = by_name.get('baseline_identical')
    if b:
        ok1 = b['score'] >= 0.90
        print(f'  1. Baseline (identical): {b["score"]:.3f} -- '
              f'{"PASS" if ok1 else "FAIL"} (need >= 0.90)')

    # 2. Cross-product negative
    c = by_name.get('cross_product_negative')
    if c:
        ok2 = c['score'] < 0.20
        print(f'  2. Cross-product (negative): {c["score"]:.3f} -- '
              f'{"PASS" if ok2 else "FAIL"} (need < 0.20)')

    # 3. Jeans gpt directional agreement with Family B
    j0 = by_name.get('jeans_gpt_iter0')
    j4 = by_name.get('jeans_gpt_iter4')
    if j0 and j4:
        delta = j4['score'] - j0['score']
        ok3 = delta > 0
        print(f'  3. Jeans gpt (iter4 > iter0 expected): '
              f'{j0["score"]:.3f} -> {j4["score"]:.3f} (delta= {delta:+.3f}) -- '
              f'{"PASS" if ok3 else "FAIL"}')

    # 4. Headphones flux directional agreement with Family B
    h0 = by_name.get('headphones_flux_iter0')
    h4 = by_name.get('headphones_flux_iter4')
    if h0 and h4:
        delta = h4['score'] - h0['score']
        ok4 = delta <= 0
        print(f'  4. Headphones flux (iter4 <= iter0 expected): '
              f'{h0["score"]:.3f} -> {h4["score"]:.3f} (delta= {delta:+.3f}) -- '
              f'{"PASS" if ok4 else "FAIL"}')

    # 5. Chess_set flux flat (truncated)
    c0 = by_name.get('chess_set_flux_iter0')
    c4 = by_name.get('chess_set_flux_iter4')
    if c0 and c4:
        delta = abs(c4['score'] - c0['score'])
        ok5 = delta < 0.10
        print(f'  5. Chess_set flux (iter0 ~ iter4 expected): '
              f'{c0["score"]:.3f} -> {c4["score"]:.3f} (|delta=| {delta:.3f}) -- '
              f'{"PASS" if ok5 else "FAIL"} (need |delta=| < 0.10)')

    print('-' * 72)
    print('INTERPRETATION')
    print('-' * 72)
    print('  - 4 or 5 checks PASS: VLM-as-judge is viable. A full v6 '
          'experiment is worth running.')
    print('  - 2 or 3 checks PASS: marginal. Scores discriminate some '
          'cases; worth trying a small-scope v6 on 2-3 pairs.')
    print('  - 0 or 1 checks PASS: VLM judge is not producing calibrated '
          'scores on this task. Skip the full experiment.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated case names to run (skip others).')
    args = parser.parse_args()

    only = args.only.split(',') if args.only else None
    results = run_spike(only=only)
    print_summary(results)

    # Save results for follow-up analysis.
    out_path = ROOT / 'exploration' / 'vlm_judge_spike_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults written to {out_path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
