'''exploration/analyze_trajectories.py

Per-iteration analysis of agent-loop trajectories for an arbitrary
config. Parametrized tool usable across any named config by passing
its name on the CLI.

Purpose
-------
Aggregate per-iteration Family B cosines (CLIP / DINOv2 / SigLIP vs
ground truth) alongside the Family C quality score and
descriptiveness / iters_taken from the per-iteration metadata, for a
named config across a set of (product, model) pairs. Three tables
emerge:

  1. Per-iteration roll-up (all {pair × iteration} rows).
  2. Iter-(last) vs iter-0 Family B delta — one row per pair, three
     numbers per row (CLIP, DINOv2, SigLIP). This is the
     "does adaptive iteration improve the image under orthogonal
     metrics" headline comparison.
  3. Candidate pairs with positive Family B slope on >=2 of 3
     encoders.

Usage
-----
  # v5 stress test (4 products, 5 iterations)
  python exploration/analyze_trajectories.py \
      --config v5_unreachable \
      --products chess_set,headphones,jeans,water_bottle \
      --iterations 0-4

Output
------
  eval_results/per_product/{config_short}_trajectories/{p}_{m}.json
    — one file per pair with full per-iteration data

  stdout
    — three tables described above; also echo CSV-like rows suitable
      for pasting into the decisions log or a slide.

Controller parameter logging gap: per-image retune values
(descriptivenessThreshold, iterStart) are not persisted. See
docs/handoff/05_decisions_log.md. This analyzer works around the
gap by surfacing the observable proxies (iters_taken per image,
final descriptivenesses per image).
'''
import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
os.chdir(_REPO_ROOT)

from PIL import Image
from eval_image import (
    clip_image_similarity,
    dinov2_similarity,
    siglip_similarity,
)
from run_post_hoc_eval import load_ground_truth_images

DEFAULT_PRODUCTS = ['backpack', 'chess_set', 'espresso_machine',
                    'headphones', 'jeans', 'water_bottle']
MODELS = ['flux', 'gpt']


def parse_iterations(spec: str) -> list[int]:
    '''Parse "0-2" or "0,1,2,3,4" into a list of ints.'''
    if '-' in spec:
        a, b = spec.split('-', 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(',')]


def short_config(config: str) -> str:
    '''v5_unreachable -> v5; vN_whatever -> vN.'''
    return config.split('_', 1)[0]


def iter_dir(product: str, model: str, config: str, iteration: int) -> str:
    return os.path.join(
        'data', product, 'trajectories',
        f'{model}_{config}', f'iteration_{iteration}',
    )


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def family_b_mean_vs_gt(cand_img, gt_images):
    return {
        'clip_img_vs_gt_mean': mean(
            [clip_image_similarity(cand_img, g) for g in gt_images]),
        'dinov2_vs_gt_mean': mean(
            [dinov2_similarity(cand_img, g) for g in gt_images]),
        'siglip_vs_gt_mean': mean(
            [siglip_similarity(cand_img, g) for g in gt_images]),
    }


def analyze_pair(product: str, model: str, config: str,
                 iterations: list[int]) -> list[dict]:
    pdir = os.path.join('data', product)
    gt_images = load_ground_truth_images(product, pdir)
    rows = []
    for i in iterations:
        idir = iter_dir(product, model, config, i)
        mpath = os.path.join(idir, 'metadata.json')
        imgpath = os.path.join(idir, 'generated_image.png')
        if not os.path.exists(mpath) or not os.path.exists(imgpath):
            print(f'  MISSING: {idir}')
            continue
        meta = json.load(open(mpath, encoding='utf-8'))
        cand = Image.open(imgpath).convert('RGB')
        fb = family_b_mean_vs_gt(cand, gt_images)
        rows.append({
            'product': product,
            'model': model,
            'config': config,
            'iteration': i,
            'descriptiveness': meta.get('descriptiveness'),
            'iters_taken': meta.get('iters_taken'),
            'score_family_c': meta.get('score'),
            **fb,
        })
    return rows


def write_pair_json(product: str, model: str, config: str,
                    rows: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f'{product}_{model}.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump({
            'product': product,
            'model': model,
            'config': config,
            'iterations': rows,
        }, f, indent=2)


def print_rollup(all_rows: list):
    print()
    print('=' * 108)
    print('Per-iteration roll-up (Family B mean cosines + Family C score + descriptiveness)')
    print('=' * 108)
    print(f'{"product":18} {"model":5} {"iter":>4}  '
          f'{"desc":>5} {"iters":>5}  '
          f'{"C":>7}  {"CLIP":>7}  {"DINOv2":>7}  {"SigLIP":>7}')
    print('-' * 108)
    for r in all_rows:
        desc = r.get('descriptiveness')
        it = r.get('iters_taken')
        c = r.get('score_family_c')
        print(f'{r["product"]:18} {r["model"]:5} {r["iteration"]:>4}  '
              f'{desc if desc is not None else "?":>5} {it if it is not None else "?":>5}  '
              f'{c if c is not None else float("nan"):>7.4f}  '
              f'{r["clip_img_vs_gt_mean"]:>7.4f}  '
              f'{r["dinov2_vs_gt_mean"]:>7.4f}  '
              f'{r["siglip_vs_gt_mean"]:>7.4f}')


def print_last_vs_first_delta(all_rows: list, products: list,
                              iterations: list[int]):
    '''Iter-(last) vs iter-0 Family B delta — the Q4-thesis headline
    comparison. One row per (product, model); three numbers per row.'''
    first, last = iterations[0], iterations[-1]
    print()
    print('=' * 104)
    print(f'Iter-{last} vs iter-{first} Family B delta  —  '
          f'headline "does adaptive iteration improve the image?" comparison')
    print('=' * 104)
    print(f'{"product":18} {"model":5}  '
          f'{f"CLIP Δ":>9} {f"DINOv2 Δ":>10} {f"SigLIP Δ":>10}  '
          f'{f"C Δ":>9}  +encoders')
    print('-' * 104)
    for p in products:
        for m in MODELS:
            trio = [r for r in all_rows
                    if r['product'] == p and r['model'] == m]
            if len(trio) < 2:
                continue
            i0 = next((r for r in trio if r['iteration'] == first), None)
            iN = next((r for r in trio if r['iteration'] == last), None)
            if i0 is None or iN is None:
                continue
            clip_d = iN['clip_img_vs_gt_mean'] - i0['clip_img_vs_gt_mean']
            dino_d = iN['dinov2_vs_gt_mean'] - i0['dinov2_vs_gt_mean']
            siglip_d = iN['siglip_vs_gt_mean'] - i0['siglip_vs_gt_mean']
            c0 = i0.get('score_family_c')
            cN = iN.get('score_family_c')
            c_d = cN - c0 if c0 is not None and cN is not None else None
            c_str = f'{c_d:+.4f}' if c_d is not None else '   n/a  '
            pos = sum(1 for d in (clip_d, dino_d, siglip_d) if d > 0.005)
            print(f'{p:18} {m:5}  '
                  f'{clip_d:+9.4f} {dino_d:+10.4f} {siglip_d:+10.4f}  '
                  f'{c_str:>9}  {pos}/3')


def print_candidates(all_rows: list, products: list,
                     iterations: list[int]):
    '''Pairs with positive Family B slope on >=2 of 3 encoders.'''
    first, last = iterations[0], iterations[-1]
    SLOPE_EPS = 0.005
    print()
    print('=' * 96)
    print(f'Candidates with positive slope on >=2 of 3 Family B metrics '
          f'(iter-{first}→iter-{last}, eps={SLOPE_EPS})')
    print('=' * 96)
    any_candidates = False
    for p in products:
        for m in MODELS:
            trio = [r for r in all_rows
                    if r['product'] == p and r['model'] == m]
            i0 = next((r for r in trio if r['iteration'] == first), None)
            iN = next((r for r in trio if r['iteration'] == last), None)
            if i0 is None or iN is None:
                continue
            deltas = [
                iN['clip_img_vs_gt_mean'] - i0['clip_img_vs_gt_mean'],
                iN['dinov2_vs_gt_mean'] - i0['dinov2_vs_gt_mean'],
                iN['siglip_vs_gt_mean'] - i0['siglip_vs_gt_mean'],
            ]
            pos = sum(1 for d in deltas if d > SLOPE_EPS)
            if pos >= 2:
                any_candidates = True
                c_d = ((iN.get('score_family_c') or 0)
                       - (i0.get('score_family_c') or 0))
                print(f'  {p}/{m}: '
                      f'CLIP{deltas[0]:+.4f}  '
                      f'DINOv2{deltas[1]:+.4f}  '
                      f'SigLIP{deltas[2]:+.4f}  '
                      f'({pos}/3 positive; Family C Δ={c_d:+.4f})')
    if not any_candidates:
        print('  (none — no pair has positive Family B slope on >=2 of 3 encoders)')


def main():
    parser = argparse.ArgumentParser(
        description='Per-iteration trajectory analysis for a named config.')
    parser.add_argument('--config', required=True,
                        help='Config name (e.g., v5_unreachable).')
    parser.add_argument('--products',
                        default=','.join(DEFAULT_PRODUCTS),
                        help='Comma-separated product slugs. '
                             'Default: all six.')
    parser.add_argument('--iterations', default='0-2',
                        help='Iteration range "A-B" or list "0,1,2". '
                             'Default: 0-2.')
    parser.add_argument('--output-dir', default=None,
                        help='Override output directory. Default: '
                             'eval_results/per_product/{short_config}_trajectories/')
    args = parser.parse_args()

    products = [p.strip() for p in args.products.split(',') if p.strip()]
    iterations = parse_iterations(args.iterations)
    out_dir = (args.output_dir
               or os.path.join('eval_results', 'per_product',
                               f'{short_config(args.config)}_trajectories'))

    print(f'Config: {args.config}')
    print(f'Products: {products}')
    print(f'Iterations: {iterations}')
    print(f'Output dir: {out_dir}')
    print()

    all_rows = []
    for p in products:
        for m in MODELS:
            print(f'[{p}/{m}] analyzing...', flush=True)
            rows = analyze_pair(p, m, args.config, iterations)
            if rows:
                write_pair_json(p, m, args.config, rows, out_dir)
                all_rows.extend(rows)

    print_rollup(all_rows)
    print_last_vs_first_delta(all_rows, products, iterations)
    print_candidates(all_rows, products, iterations)


if __name__ == '__main__':
    main()
