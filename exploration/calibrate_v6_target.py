'''
v6 quality_target calibration.

Computes VLM judge scores on iter_0 images across the 8 canonical v6
pairs (backpack/chess_set/jeans/water_bottle x flux/gpt). Uses v3
iter_0 images as a proxy -- same initial prompt and default controller
params mean v6 iter_0 will produce effectively the same image, so the
baseline VLM judge distribution transfers.

Output: per-pair integer scores and recommended quality_target for v6.

Rule: target = max(iter_0 scores) + 0.05, clamped to [0.60, 0.95].

Usage:
    python exploration/calibrate_v6_target.py
'''

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv(ROOT / '.env')

from eval_image import vlm_judge_eval


CANONICAL_PAIRS = [
    ('backpack', 'flux'),
    ('backpack', 'gpt'),
    ('chess_set', 'flux'),
    ('chess_set', 'gpt'),
    ('jeans', 'flux'),
    ('jeans', 'gpt'),
    ('water_bottle', 'flux'),
    ('water_bottle', 'gpt'),
]


def iter0_image_path(slug: str, model: str) -> Path:
    '''v3 iter_0 image serves as the proxy. Same initial prompt + default
    controller params as v6 iter_0 will use.'''
    return (ROOT / 'data' / slug / 'trajectories'
            / f'{model}_v3_initial_prompt_features' / 'iteration_0'
            / 'generated_image.png')


def gt_image_path(slug: str) -> Path:
    return ROOT / 'data' / slug / 'images' / 'main.jpg'


def main():
    print('Computing VLM judge iter_0 scores for v6 calibration.')
    print('Using v3 iter_0 images as proxy (same initial prompt, same')
    print('default controller params -> same iter_0 image as v6 will produce).')
    print()

    scores = []
    for slug, model in CANONICAL_PAIRS:
        gen = iter0_image_path(slug, model)
        gt = gt_image_path(slug)

        if not gen.exists():
            print(f'  {slug:<14} {model:<4}  MISSING: {gen}')
            continue

        try:
            score = vlm_judge_eval(str(gen), str(gt), slug=slug)
            int_score = round(score * 100)
            scores.append((slug, model, score, int_score))
            print(f'  {slug:<14} {model:<4}  iter_0 VLM judge = {int_score} ({score:.3f})')
        except Exception as e:
            print(f'  {slug:<14} {model:<4}  ERROR: {e}')

    if not scores:
        print('\nNo scores computed. Aborting.')
        sys.exit(1)

    vals = [s[2] for s in scores]
    min_s = min(vals)
    median_s = sorted(vals)[len(vals) // 2]
    max_s = max(vals)

    print()
    print(f'  min:    {min_s:.3f}  ({round(min_s*100)})')
    print(f'  median: {median_s:.3f}  ({round(median_s*100)})')
    print(f'  max:    {max_s:.3f}  ({round(max_s*100)})')
    print(f'  spread: {max_s - min_s:.3f}')

    raw_target = max_s + 0.05
    target = max(0.60, min(0.95, raw_target))
    print()
    print(f'  raw target (max + 0.05):    {raw_target:.3f}')
    print(f'  clamped to [0.60, 0.95]:    {target:.3f}')
    print(f'  integer equivalent:         {round(target*100)}')

    # Save results for the record.
    out = {
        'per_pair_scores': [
            {'slug': s, 'model': m, 'score_float': sc, 'score_int': i}
            for s, m, sc, i in scores
        ],
        'distribution': {
            'min': min_s, 'median': median_s, 'max': max_s,
            'spread': max_s - min_s,
        },
        'recommended_target': target,
        'target_rule': 'max + 0.05, clamped to [0.60, 0.95]',
    }
    out_path = ROOT / 'exploration' / 'calibrate_v6_target_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults written to {out_path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
