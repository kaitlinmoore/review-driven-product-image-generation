'''extract_iteration_trajectories.py
---------------------------------
Copies per-iteration artifacts (prompt + generated image + scoring metadata)
out of the timestamped `runs/` subdirectories into clean, navigable
per-(product, model, config) folders under `data/{product}/trajectories/`.

Lets writeup and slide-deck teams browse every iteration for any outcome
without needing to figure out timestamped directory names. Also solves
the "where do iterations live?" problem when `runs/` is gitignored.

For each product:
  - Find every agent_run_{model}_{config}_meta.json file
  - Read its `run_dir` field (the timestamped path under runs/)
  - Copy each iteration_i/ subdirectory to
    `data/{product}/trajectories/{model}_{config}/iteration_{i}/`
  - Extract the prompt text from each iteration's metadata.json into a
    standalone `prompt.txt` for easy browsing on GitHub

Output per iteration:
    data/{product}/trajectories/{model}_{config}/iteration_{i}/
        prompt.txt           | the refined prompt used for this iteration
        generated_image.png  | the image produced
        metadata.json        | score, descriptiveness, iters_taken, prompt

Idempotent: re-running skips iterations whose target already exists unless
--force is passed.

Usage and CLI Flags:
    python src/extract_iteration_trajectories.py              # all 6 products × 2 models × 3 configs
    python src/extract_iteration_trajectories.py --only water_bottle
    python src/extract_iteration_trajectories.py --force      # overwrite existing
'''

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

# Try to prevent encoding errors.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


## CONSTANTS ##

DATA_DIR = 'data'

PRODUCTS = ['backpack', 'chess_set', 'espresso_machine',
            'headphones', 'jeans', 'water_bottle']


## HELPERS ##

def parse_meta_filename(filename: str) -> tuple[str, str]:
    '''Extract (model_tag, config_name) from a filename like
    `agent_run_flux_v1_title_clip_meta.json`.'''
    base = os.path.basename(filename).replace('_meta.json', '').replace('agent_run_', '')
    # base is now like 'flux_v1_title_clip' or 'gpt_v3_initial_prompt_features'
    parts = base.split('_', 1)
    if len(parts) != 2:
        raise ValueError(f'Unexpected meta filename shape: {filename}')
    return parts[0], parts[1]   # (model, config)


def copy_iteration(src_dir: str, dst_dir: str, force: bool) -> str:
    '''Copy one iteration_i/ subtree, extract prompt.txt from metadata.json.
    Returns 'copied', 'skipped_exists', or 'skipped_missing'.'''
    if not os.path.isdir(src_dir):
        return 'skipped_missing'
    if os.path.isdir(dst_dir) and not force:
        return 'skipped_exists'

    if os.path.isdir(dst_dir) and force:
        shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir)

    # Extract the prompt into a standalone text file for easier browsing.
    meta_path = os.path.join(dst_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        prompt = meta.get('prompt', '').strip()
        if prompt:
            with open(os.path.join(dst_dir, 'prompt.txt'), 'w', encoding='utf-8') as f:
                f.write(prompt + '\n')

    return 'copied'


## DRIVER ##

def process_product(slug: str, force: bool) -> dict:
    pdir = os.path.join(DATA_DIR, slug)
    if not os.path.isdir(pdir):
        return {'slug': slug, 'error': f'product dir not found: {pdir}'}

    meta_pattern = os.path.join(pdir, 'agent_run_*_meta.json')
    meta_files = sorted(glob.glob(meta_pattern))
    if not meta_files:
        return {'slug': slug, 'skipped': True,
                'reason': 'no agent_run_*_meta.json files'}

    summary = {'slug': slug, 'copied': 0, 'skipped_exists': 0,
                'skipped_missing': 0, 'combos': 0}

    for meta_file in meta_files:
        try:
            model_tag, config_name = parse_meta_filename(meta_file)
        except ValueError:
            print(f'  [!] skipping unparseable filename: {meta_file}')
            continue

        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        run_dir = meta.get('run_dir')
        image_count = meta.get('image_count', 3)
        if not run_dir:
            print(f'  [!] {model_tag}/{config_name}: no run_dir in meta')
            continue
        if not os.path.isdir(run_dir):
            print(f'  [!] {model_tag}/{config_name}: run_dir missing '
                  f'({run_dir}) — can\'t extract iterations')
            continue

        target_parent = os.path.join(pdir, 'trajectories',
                                      f'{model_tag}_{config_name}')
        os.makedirs(target_parent, exist_ok=True)
        summary['combos'] += 1

        for i in range(image_count):
            src_iter = os.path.join(run_dir, f'iteration_{i}')
            dst_iter = os.path.join(target_parent, f'iteration_{i}')
            result = copy_iteration(src_iter, dst_iter, force)
            summary[result] = summary.get(result, 0) + 1

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Extract per-iteration trajectory artifacts from '
                    'timestamped runs/ subdirs to clean per-(product, model, '
                    'config) folders under data/{product}/trajectories/.')
    parser.add_argument('--only', default=None,
                        help='Substring-match a single product slug.')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing iteration folders.')
    args = parser.parse_args()

    products = PRODUCTS
    if args.only:
        needle = args.only.lower()
        products = [p for p in PRODUCTS if needle in p.lower()]
        if not products:
            print(f'[!] --only {args.only!r} matches no product.')
            sys.exit(1)

    print(f'Processing {len(products)} product(s): {products}\n')
    totals = {'copied': 0, 'skipped_exists': 0, 'skipped_missing': 0, 'combos': 0}

    for slug in products:
        print(f'[{slug}]')
        r = process_product(slug, args.force)
        if r.get('error'):
            print(f'  [!] {r["error"]}')
            continue
        if r.get('skipped'):
            print(f'  skipped: {r["reason"]}')
            continue
        print(f'  combos: {r["combos"]}  '
              f'copied: {r["copied"]}  '
              f'skipped_exists: {r["skipped_exists"]}  '
              f'skipped_missing: {r["skipped_missing"]}')
        for k in ('copied', 'skipped_exists', 'skipped_missing', 'combos'):
            totals[k] += r.get(k, 0)

    print()
    print('=' * 60)
    print(f'TOTAL: {totals["combos"]} (model, config) combinations  '
          f'{totals["copied"]} iterations copied  '
          f'{totals["skipped_exists"]} skipped (existed)  '
          f'{totals["skipped_missing"]} skipped (source missing)')


if __name__ == '__main__':
    main()
