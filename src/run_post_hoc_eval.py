'''run_post_hoc_eval.py
--------------------
Post-hoc evaluation pass over Phase 3 and Phase 4 outputs. For every
(product, image-model, config) combination, computes:

  Family B | Image-vs-image cosines (CLIP, DINOv2, SigLIP):
            generated image vs each non-excluded ground-truth image,
            aggregated as mean and max across reference images.

  Family C | Structured-feature per-field agreement:
            both generated-image features AND converged-prompt features,
            against both the initial-prompt feature reference (preservation
            check) AND the ground-truth feature reference (accuracy check).

In-loop quality numbers (best_q, best_idx, descriptiveness/quality/iter
trajectories) are pulled directly from agent_run_*_meta.json and copied
into the summary so a downstream reader has all the headline numbers in
one row per combination.

Outputs:
    eval_results/summary.csv             | wide table, 36 rows × ~25 cols
    eval_results/per_product/{slug}.json | per-product detail with per-field
                                           Family C breakdown for failure-mode
                                           analysis

Replay-aware: every underlying eval_image function is cached. First
run takes ~3-5 minutes (CLIP/DINOv2/SigLIP image embeddings are computed
fresh because the agent loop only used image-vs-text CLIP, not these
image-vs-image variants). Subsequent runs are instant.

Usage and CLI Flags:
    python src/run_post_hoc_eval.py            # all 36 combinations
    python src/run_post_hoc_eval.py --only water_bottle   # one product
'''

import argparse
import csv
import json
import os
import sys
import time

from pathlib import Path
from PIL import Image

from eval_image import (
    clip_image_similarity,
    dinov2_similarity,
    siglip_similarity,
    feature_agreement,
    per_field_agreement,
)
from extract_structured_features import _list_product_images

# Try to prevent encoding errors.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


## CONSTANTS ##

DATA_DIR = 'data'
OUT_DIR = 'eval_results'

PRODUCTS = ['backpack', 'chess_set', 'espresso_machine',
            'headphones', 'jeans', 'water_bottle']
MODELS = ['flux', 'gpt']
CONFIGS = ['v1_title_clip', 'v2_initial_prompt_clip',
           'v3_initial_prompt_features',
           'v5_unreachable']

SCHEMA_FIELDS = (
    'product_type', 'primary_colors', 'materials', 'finish', 'texture',
    'size_descriptor', 'measurements', 'shape_and_form',
    'decorative_elements', 'visible_parts',
    'brand_visibility', 'brand_description', 'overall_aesthetic',
)


## HELPERS ##

def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def aggregate(scores: list[float]) -> tuple[float, float]:
    '''Return (mean, max). Empty list returns (0.0, 0.0).'''
    if not scores:
        return (0.0, 0.0)
    return (sum(scores) / len(scores), max(scores))


def load_ground_truth_images(slug: str, pdir: str) -> list[Image.Image]:
    '''PIL.Image objects for every non-excluded ground-truth photo for a product.'''
    images_dir = os.path.join(pdir, 'images')
    paths = _list_product_images(slug, images_dir)
    return [Image.open(p).convert('RGB') for p in paths]


def family_b_for_combo(
    generated_image_path: str,
    gt_images: list[Image.Image],
) -> dict:
    '''Family B: image-vs-image cosines aggregated across ground-truth photos.
    Returns a flat dict suitable for the summary CSV.'''
    if not os.path.exists(generated_image_path):
        return {
            'clip_img_vs_gt_mean': None, 'clip_img_vs_gt_max': None,
            'dinov2_vs_gt_mean': None, 'dinov2_vs_gt_max': None,
            'siglip_vs_gt_mean': None, 'siglip_vs_gt_max': None,
            'gen_image_present': False,
        }
    gen_img = Image.open(generated_image_path).convert('RGB')

    clip_scores = [clip_image_similarity(gen_img, gt) for gt in gt_images]
    dinov2_scores = [dinov2_similarity(gen_img, gt) for gt in gt_images]
    siglip_scores = [siglip_similarity(gen_img, gt) for gt in gt_images]

    clip_mean, clip_max = aggregate(clip_scores)
    dino_mean, dino_max = aggregate(dinov2_scores)
    siglip_mean, siglip_max = aggregate(siglip_scores)

    return {
        'clip_img_vs_gt_mean': clip_mean,
        'clip_img_vs_gt_max': clip_max,
        'dinov2_vs_gt_mean': dino_mean,
        'dinov2_vs_gt_max': dino_max,
        'siglip_vs_gt_mean': siglip_mean,
        'siglip_vs_gt_max': siglip_max,
        'clip_per_image': clip_scores,
        'dinov2_per_image': dinov2_scores,
        'siglip_per_image': siglip_scores,
        'gen_image_present': True,
    }


def family_c_for_combo(
    generated_features_path: str,
    converged_features_path: str,
    initial_features: dict,
    ground_truth_features: dict,
) -> dict:
    '''Family C: per-field structured-feature agreement against initial and
    ground-truth references. Returns the overall scores plus per-field
    breakdowns for both reference comparisons.'''
    out = {}

    # generated features → initial / ground_truth
    if os.path.exists(generated_features_path):
        gen_features = load_json(generated_features_path)
        gen_vs_initial = per_field_agreement(gen_features, initial_features)
        gen_vs_gt = per_field_agreement(gen_features, ground_truth_features)
        out.update({
            'gen_features_vs_initial': gen_vs_initial['overall'],
            'gen_features_vs_ground_truth': gen_vs_gt['overall'],
            'gen_per_field_vs_initial': gen_vs_initial['per_field'],
            'gen_per_field_vs_ground_truth': gen_vs_gt['per_field'],
            'gen_features_present': True,
        })
    else:
        out.update({
            'gen_features_vs_initial': None,
            'gen_features_vs_ground_truth': None,
            'gen_per_field_vs_initial': None,
            'gen_per_field_vs_ground_truth': None,
            'gen_features_present': False,
        })

    # converged features → initial / ground_truth
    if os.path.exists(converged_features_path):
        conv_features = load_json(converged_features_path)
        conv_vs_initial = per_field_agreement(conv_features, initial_features)
        conv_vs_gt = per_field_agreement(conv_features, ground_truth_features)
        out.update({
            'conv_features_vs_initial': conv_vs_initial['overall'],
            'conv_features_vs_ground_truth': conv_vs_gt['overall'],
            'conv_per_field_vs_initial': conv_vs_initial['per_field'],
            'conv_per_field_vs_ground_truth': conv_vs_gt['per_field'],
            'conv_features_present': True,
        })
    else:
        out.update({
            'conv_features_vs_initial': None,
            'conv_features_vs_ground_truth': None,
            'conv_per_field_vs_initial': None,
            'conv_per_field_vs_ground_truth': None,
            'conv_features_present': False,
        })

    return out


def in_loop_metrics(meta_path: str) -> dict:
    '''Pull the headline in-loop trajectory numbers from agent_run_*_meta.json.'''
    if not os.path.exists(meta_path):
        return {
            'in_loop_best_q': None, 'in_loop_best_idx': None,
            'iters_taken_per_image': None, 'qualities': None,
            'descriptivenesses': None,
            'agent_run_present': False,
        }
    meta = load_json(meta_path)
    return {
        'in_loop_best_q': meta.get('qualities', [0])[meta.get('best_idx', 0)]
                          if meta.get('qualities') else None,
        'in_loop_best_idx': meta.get('best_idx'),
        'iters_taken_per_image': meta.get('iters_taken_per_image'),
        'qualities': meta.get('qualities'),
        'descriptivenesses': meta.get('descriptivenesses'),
        'agent_run_present': True,
    }


## DRIVER ##

def evaluate_combo(slug: str, pdir: str, model_tag: str, config_name: str,
                    initial_features: dict, gt_features: dict,
                    gt_images: list[Image.Image]) -> dict:
    '''Compute all metrics for one (product, model, config) combination.'''
    gen_image_path = os.path.join(
        pdir, f'generated_image_{model_tag}_{config_name}.png')
    gen_features_path = os.path.join(
        pdir, f'structured_features_generated_{model_tag}_{config_name}_v1.json')
    conv_features_path = os.path.join(
        pdir, f'structured_features_converged_{model_tag}_{config_name}_v1.json')
    meta_path = os.path.join(
        pdir, f'agent_run_{model_tag}_{config_name}_meta.json')

    fb = family_b_for_combo(gen_image_path, gt_images)
    fc = family_c_for_combo(gen_features_path, conv_features_path,
                              initial_features, gt_features)
    inloop = in_loop_metrics(meta_path)

    return {
        'product': slug,
        'model': model_tag,
        'config': config_name,
        **inloop,
        **fb,
        **fc,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute Family B + C ground-truth comparisons across '
                    'every (product, model, config) combination.')
    parser.add_argument('--only', default=None,
                        help='Substring-match a single product slug.')
    parser.add_argument('--out-dir', default=OUT_DIR,
                        help=f'Output directory. Default {OUT_DIR}.')
    args = parser.parse_args()

    products_to_eval = PRODUCTS
    if args.only:
        needle = args.only.lower()
        products_to_eval = [p for p in PRODUCTS if needle in p.lower()]
        if not products_to_eval:
            print(f'[!] --only {args.only!r} matches no product.')
            sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_product_dir = out_dir / 'per_product'
    per_product_dir.mkdir(parents=True, exist_ok=True)

    print(f'Evaluating {len(products_to_eval)} product(s) '
          f'× {len(MODELS)} models × {len(CONFIGS)} configs '
          f'= {len(products_to_eval) * len(MODELS) * len(CONFIGS)} combinations\n')

    all_rows = []
    t0 = time.time()

    for slug in products_to_eval:
        pdir = os.path.join(DATA_DIR, slug)
        print(f'[{slug}]')

        # Load per-product reference data once and reuse across combos.
        initial_features = load_json(
            os.path.join(pdir, 'structured_features_initial_v1.json'))
        gt_features = load_json(
            os.path.join(pdir, 'structured_features_ground_truth_v1.json'))
        gt_images = load_ground_truth_images(slug, pdir)
        print(f'  ground-truth images: {len(gt_images)}')

        product_results = {
            'slug': slug,
            'n_ground_truth_images': len(gt_images),
            'configs': {},
        }

        for model_tag in MODELS:
            for config_name in CONFIGS:
                t_combo = time.time()
                row = evaluate_combo(
                    slug, pdir, model_tag, config_name,
                    initial_features, gt_features, gt_images)
                elapsed = time.time() - t_combo
                all_rows.append(row)

                # Per-product detail JSON: include the rich per-field breakdowns
                # that the wide CSV doesn't capture.
                product_results['configs'][f'{model_tag}_{config_name}'] = {
                    k: row[k] for k in row
                    if k not in ('product', 'model', 'config')
                }

                if row['gen_image_present']:
                    print(f'  {model_tag}/{config_name}: '
                          f'in_loop_best_q={row["in_loop_best_q"]:.4f}  '
                          f'gen_vs_gt_clip_mean={row["clip_img_vs_gt_mean"]:.4f}  '
                          f'gen_vs_gt_features={row["gen_features_vs_ground_truth"]:.4f}  '
                          f'({elapsed:.1f}s)')
                else:
                    print(f'  {model_tag}/{config_name}: '
                          f'(generated image missing — skipped)')

        # Write per-product JSON.
        out_path = per_product_dir / f'{slug}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(product_results, f, indent=2, ensure_ascii=False)
        print(f'  wrote {out_path}')
        print()

    # Write summary CSV: one row per (product, model, config), wide format.
    # Drop the per-image and per-field detail (those live in per_product JSONs).
    flat_columns = (
        'product', 'model', 'config',
        'in_loop_best_q', 'in_loop_best_idx',
        'clip_img_vs_gt_mean', 'clip_img_vs_gt_max',
        'dinov2_vs_gt_mean', 'dinov2_vs_gt_max',
        'siglip_vs_gt_mean', 'siglip_vs_gt_max',
        'gen_features_vs_initial', 'gen_features_vs_ground_truth',
        'conv_features_vs_initial', 'conv_features_vs_ground_truth',
    )
    summary_path = out_dir / 'summary.csv'
    with open(summary_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=flat_columns, extrasaction='ignore')
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f'wrote {summary_path}  ({len(all_rows)} rows)')
    print(f'\nTotal: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
