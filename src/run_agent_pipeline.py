'''run_agent_pipeline.py
---------------------
Per-product × per-image-model driver for the agent loop.

Composition layer that iterates products discovered under data/, builds a
reviews DataLoader, calls agentLoop with the right hyperparameters, and
persists per-config, per-product, per-model artifacts.

Artifact layout:

    Per-config (always written):
        data/{product}/converged_prompt_{model}_{config}.txt
        data/{product}/generated_image_{model}_{config}.png
        data/{product}/agent_run_{model}_{config}_meta.json

    Canonical pointers (written unless --no-promote):
        data/{product}/converged_prompt_{model}.txt
        data/{product}/generated_image_{model}.png
        data/{product}/converged_prompt.txt           # = converged_prompt_flux (canonical for
                                                        extract_structured_features --source converged)

The per-config files are the true artifacts; the canonical pointers are
convenience aliases that downstream stages (extract_structured_features,
any notebook / ad-hoc analysis) can depend on without needing to know
which config is currently "blessed". Running a second config with
--no-promote lets you preserve the first config as canonical while still
capturing the ablation.

A fresh PromptWriter is created per (product, model) pair so model 2 does not
start from model 1's already-refined prompt. The dataloader is shared across
both model runs for the same product so both models see identical review
inputs — isolating "which model renders better" from "which trajectory saw
which reviews".

Usage and CLI flags (works in both PowerShell and bash):
    # v1: CLIP cosine vs product title (default — original config)
    python src/run_agent_pipeline.py --config-name v1_title_clip_eval

    # v2: CLIP cosine vs initial_prompt.txt content
    python src/run_agent_pipeline.py --config-name v2_initial_prompt_clip \
        --quality-signal clip_text --reference initial_prompt

    # v3: structured-feature agreement vs initial_prompt's pre-extracted features
    python src/run_agent_pipeline.py --config-name v3_initial_prompt_features \
        --quality-signal structured_features --reference initial_prompt

    # Run an ablation without overwriting canonical pointers
    python src/run_agent_pipeline.py --config-name v3_initial_prompt_features \
        --quality-signal structured_features --reference initial_prompt --no-promote

    # Restrict to one product / model / smaller iteration count for smoke-test
    python src/run_agent_pipeline.py --config-name v2_initial_prompt_clip \
        --quality-signal clip_text --reference initial_prompt \
        --only water_bottle --model 1 --image-count 2

Config-name is required (no default) — every run stamps its hyperparameter /
signal choices into a named config so two runs with different designs
remain distinguishable on disk and in the meta JSON.

Replay awareness: all the expensive calls (Mistral stepPrompt / ratePrompt,
FLUX.1-schnell generation, gpt-image-1.5 generation, CLIP evaluation) are
wrapped with cached_call downstream. Set REPLAY_MODE=replay to rerun from
the committed cache with no live calls.

    # PowerShell
    $env:REPLAY_MODE = 'replay'
    python src/run_agent_pipeline.py --config-name v1_title_clip_eval

    # bash / zsh
    REPLAY_MODE=replay python src/run_agent_pipeline.py --config-name v1_title_clip_eval
'''

import argparse
import datetime
import json
import os
import shutil
import sys
import time

# Use for local .env defined API keys/tokens.
from dotenv import load_dotenv

from agent_loop import agentLoop
from eval_image import compImage, eval_structured_features
from prompt_writer import load_mistral
from reviews_dataloader import make_reviews_dataloader

# Try to prevent encoding errors.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


## CONSTANTS ##

DATA_DIR = 'data'

# Model-num -> filename tag. Keeps filenames human-readable instead of
# `generated_image_1.png` / `generated_image_2.png`.
MODEL_TAGS = {1: 'flux', 2: 'gpt'}

# Hyperparameter defaults. Tune at runtime via CLI flags; the numbers below
# are starting points, not analytical optima.
DEFAULT_DESCRIPTIVENESS_THRESHOLD = 60.0
DEFAULT_ITER_START = 3
DEFAULT_ITER_MIN = 1
DEFAULT_ITER_MAX = 5
DEFAULT_IMAGE_COUNT = 3          # >=3 so the quadratic retune has enough points
DEFAULT_QUALITY_TARGET = 0.5     # canonical setting used in v1/v2 (CLIP image-vs-text cosine)


## DISCOVERY ##

def discover_products() -> dict[str, str]:
    '''Products with the inputs the agent loop needs:
        metadata.json (for product title used as v1 reference text)
        initial_prompt.txt (for PromptWriter seed)
        reviews_ranked.jsonl (for the DataLoader)
    '''
    out: dict[str, str] = {}
    if not os.path.isdir(DATA_DIR):
        return out
    for name in sorted(os.listdir(DATA_DIR)):
        pdir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(pdir) or name == 'filter_caches':
            continue
        needed = ('metadata.json', 'initial_prompt.txt', 'reviews_ranked.jsonl')
        if all(os.path.exists(os.path.join(pdir, f)) for f in needed):
            out[name] = pdir
    return out


def resolve_only(products: dict[str, str], needle: str) -> dict[str, str]:
    matches = [k for k in products if needle.lower() in k.lower()]
    if len(matches) == 0:
        print(f'[!] --only {needle!r} matches no product. Available: {sorted(products)}')
        sys.exit(1)
    if len(matches) > 1:
        print(f'[!] --only {needle!r} matches multiple: {matches}.')
        sys.exit(1)
    return {matches[0]: products[matches[0]]}


def resolve_models(flag: str) -> list[int]:
    '''--model accepts '1', '2', or 'both'. Returns a list in fixed order
    [1, 2] for 'both' so FLUX (reproducible, open-weights) runs first.'''
    if flag == '1':
        return [1]
    if flag == '2':
        return [2]
    if flag == 'both':
        return [1, 2]
    raise ValueError(f'--model must be 1, 2, or both; got {flag!r}')


## PER-PRODUCT × PER-MODEL DRIVER ##

def per_config_paths(pdir: str, model_tag: str, config_name: str) -> dict[str, str]:
    '''Paths for the per-(product, model, config) artifact triplet.'''
    return {
        'converged': os.path.join(pdir, f'converged_prompt_{model_tag}_{config_name}.txt'),
        'image': os.path.join(pdir, f'generated_image_{model_tag}_{config_name}.png'),
        'meta': os.path.join(pdir, f'agent_run_{model_tag}_{config_name}_meta.json'),
    }


def canonical_paths(pdir: str, model_tag: str) -> dict[str, str]:
    '''Paths for the canonical pointer artifacts that downstream stages
    (extract_structured_features, ad-hoc analysis) read with no knowledge
    of which config is blessed.'''
    return {
        'converged': os.path.join(pdir, f'converged_prompt_{model_tag}.txt'),
        'image': os.path.join(pdir, f'generated_image_{model_tag}.png'),
        # Global (model-agnostic) converged prompt. extract_structured_features
        # reads this when invoked as `--source converged` with no flags.
        'converged_global': os.path.join(pdir, 'converged_prompt.txt'),
    }


def artifacts_complete(pdir: str, model_tag: str, config_name: str) -> bool:
    '''True iff the per-config (product, model, config) triplet is all present.
    Canonical pointers are ignored — they are idempotently rewritten per run.'''
    p = per_config_paths(pdir, model_tag, config_name)
    return os.path.exists(p['converged']) and os.path.exists(p['image']) \
        and os.path.exists(p['meta'])


def read_product_title(pdir: str) -> str:
    '''Product title from metadata.json. Title-level signal only — no image
    bytes or features are used as a refinement signal.'''
    with open(os.path.join(pdir, 'metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    title = (metadata.get('title') or '').strip()
    if not title:
        raise ValueError(f'metadata.json at {pdir} has empty title')
    return title


def read_initial_prompt(pdir: str) -> str:
    '''The free-form visual description from initial_prompt.txt.
    ~250 words distilled from metadata + filtered reviews by gpt-4o.'''
    path = os.path.join(pdir, 'initial_prompt.txt')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def read_initial_features(pdir: str) -> dict:
    '''Pre-extracted 13-field structured-features dict from
    structured_features_initial_v1.json. The reference for v3.'''
    path = os.path.join(pdir, 'structured_features_initial_v1.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_quality_signal_fn(quality_signal: str, reference: str,
                             pdir: str, slug: str):
    '''Return (quality_signal_fn, reference_summary).

    quality_signal_fn: Callable[[PIL.Image], float] passed to agentLoop.
    reference_summary: short string stamped into the meta JSON for traceability
                       (e.g., the actual title text or path-of-reference).
    slug: product slug — passed through to eval_structured_features so its
          gpt-4o vision-extraction cache shares with the post-hoc
          `extract_structured_features.py --source generated` flow.'''
    if quality_signal == 'clip_text':
        if reference == 'title':
            ref_text = read_product_title(pdir)
            ref_summary = f'title: {ref_text[:100]}'
        elif reference == 'initial_prompt':
            ref_text = read_initial_prompt(pdir)
            ref_summary = f'initial_prompt.txt ({len(ref_text)} chars)'
        else:
            raise ValueError(
                f'--quality-signal=clip_text does not support --reference={reference!r}')
        return (lambda img: compImage(img, ref_text)), ref_summary

    if quality_signal == 'structured_features':
        if reference == 'initial_prompt':
            ref_features = read_initial_features(pdir)
            ref_summary = 'structured_features_initial_v1.json'
        else:
            raise ValueError(
                f'--quality-signal=structured_features does not support --reference={reference!r}')
        return (lambda img: eval_structured_features(img, ref_features, slug=slug)), ref_summary

    raise ValueError(f'unknown --quality-signal: {quality_signal!r}')


def run_one(slug: str, pdir: str, image_model_num: int,
            dataloader, model, tokenizer, args) -> dict:
    '''Run agentLoop for one (product, image-model) pair and persist per-config
    artifacts. Also updates canonical pointers unless --no-promote is set.
    Returns a summary dict.'''
    tag = MODEL_TAGS[image_model_num]
    config = args.config_name

    if artifacts_complete(pdir, tag, config) and not args.force:
        print(f'  [{slug}/{tag}/{config}] artifacts already present — skipping (use --force).')
        return {'slug': slug, 'model_tag': tag, 'config_name': config, 'skipped': True}

    initial_prompt_path = os.path.join(pdir, 'initial_prompt.txt')

    # Build the per-(product) quality-signal closure based on --quality-signal
    # and --reference. agentLoop is signal-agnostic; the driver decides which
    # metric and reference to use. The product slug is threaded through so
    # in-loop vision-extraction caches share with --source generated.
    quality_signal_fn, reference_summary = build_quality_signal_fn(
        args.quality_signal, args.reference, pdir, slug)

    t0 = time.time()
    try:
        (itersTaken, descriptivenesses, prompts, images, qualities,
         run_dir) = agentLoop(
            dataloader=dataloader,
            model=model,
            tokenizer=tokenizer,
            quality_signal_fn=quality_signal_fn,
            descriptivenessThreshold=args.descriptiveness_threshold,
            iterStart=args.iter_start,
            iterMin=args.iter_min,
            iterMax=args.iter_max,
            imageCount=args.image_count,
            imageModelNum=image_model_num,
            qualityTarget=args.quality_target,
            initial_prompt_path=initial_prompt_path,
        )
    except Exception as e:
        print(f'  [{slug}/{tag}/{config}] [!] agentLoop failed: {type(e).__name__}: {e}')
        return {'slug': slug, 'model_tag': tag, 'config_name': config, 'error': str(e)}

    elapsed = time.time() - t0

    # Canonical pair = the iteration whose image scored best under evalImage.
    # Pairing prompt and image from the same iteration keeps text <-> image
    # alignment (the saved prompt is the one that actually produced the saved
    # image). prompts[-1] (the most-refined trajectory endpoint) is also
    # recorded in the meta JSON for reference.
    best_idx = max(range(len(qualities)), key=lambda i: qualities[i])

    # Per-config artifact paths: these are the true record for this run.
    pc = per_config_paths(pdir, tag, config)

    # Atomic write of the per-config converged prompt text.
    tmp = pc['converged'] + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(prompts[best_idx] + '\n')
    os.replace(tmp, pc['converged'])

    # Copy the best-iteration image from runs/ into the per-config data/ path.
    src_image = os.path.join(run_dir, f'iteration_{best_idx}', 'generated_image.png')
    if not os.path.exists(src_image):
        # Defensive — should never fire since agentLoop just wrote it.
        raise FileNotFoundError(f'expected per-iteration image missing: {src_image}')
    shutil.copyfile(src_image, pc['image'])

    # Provenance sidecar. All hyperparameters + observed trajectory + the
    # config name so a replay attempt can verify it matches the canonical record.
    meta = {
        'slug': slug,
        'image_model_num': image_model_num,
        'model_tag': tag,
        'config_name': config,
        'quality_signal': args.quality_signal,
        'reference': args.reference,
        'reference_summary': reference_summary,
        'image_count': args.image_count,
        'descriptiveness_threshold': args.descriptiveness_threshold,
        'iter_start': args.iter_start,
        'iter_min': args.iter_min,
        'iter_max': args.iter_max,
        'quality_target': args.quality_target,
        'iters_taken_per_image': itersTaken,
        'descriptivenesses': descriptivenesses,
        'qualities': qualities,
        'best_idx': best_idx,
        'final_prompt_is_best': best_idx == len(prompts) - 1,
        'run_dir': run_dir,
        'elapsed_seconds': round(elapsed, 2),
        'timestamp_utc': datetime.datetime.now(datetime.UTC).isoformat(
            timespec='seconds').replace('+00:00', 'Z'),
    }
    with open(pc['meta'], 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Promote to canonical pointers unless --no-promote. These are what
    # downstream stages read by default; skipping promotion preserves the
    # previously-promoted config as canonical while still capturing this run
    # as a named ablation.
    promoted = False
    if not args.no_promote:
        can = canonical_paths(pdir, tag)
        shutil.copyfile(pc['converged'], can['converged'])
        shutil.copyfile(pc['image'], can['image'])
        # Global converged_prompt.txt is the FLUX-only convention (FLUX is
        # the open-weights reproducible model). Only update it from the FLUX
        # run; leave it alone for the GPT run so we don't overwrite.
        if tag == 'flux':
            shutil.copyfile(pc['converged'], can['converged_global'])
        promoted = True

    promo_label = 'promoted' if promoted else 'not-promoted'
    print(f'  [{slug}/{tag}/{config}] best_idx={best_idx} q={qualities[best_idx]:.4f}  '
          f'iters={itersTaken} descr={[round(d,1) for d in descriptivenesses]}  '
          f'{elapsed:.1f}s  ({promo_label})')
    return {'slug': slug, 'model_tag': tag, 'promoted': promoted, **meta}


## MAIN ##

def main():
    parser = argparse.ArgumentParser(
        description='Run the agent loop per product × per image model.')
    parser.add_argument('--config-name', required=True,
                        help='Named config tag for this run (e.g. '
                             '"v1_title_clip_eval"). Stamped into artifact filenames '
                             'and into agent_run_*_meta.json["config_name"]. Required '
                             '(no default) so every run self-identifies.')
    parser.add_argument('--no-promote', action='store_true',
                        help='Write per-config artifacts only. Leaves the canonical '
                             'pointer files (converged_prompt.txt, generated_image_*.png) '
                             'untouched. Use for ablations that should not replace '
                             'the blessed config.')
    parser.add_argument('--only', default=None,
                        help='Substring-match a single product slug.')
    parser.add_argument('--model', choices=['1', '2', 'both'], default='both',
                        help='Image model(s) to run. 1=FLUX, 2=gpt-image. Default: both.')
    parser.add_argument('--force', action='store_true',
                        help='Redo even if per-config artifacts already exist.')
    parser.add_argument('--image-count', type=int, default=DEFAULT_IMAGE_COUNT,
                        help=f'Images per (product, model) run. Default {DEFAULT_IMAGE_COUNT}.')
    parser.add_argument('--iter-start', type=int, default=DEFAULT_ITER_START,
                        help=f'Starting iterMax for the first image. Default {DEFAULT_ITER_START}.')
    parser.add_argument('--iter-min', type=int, default=DEFAULT_ITER_MIN,
                        help=f'Floor on the retuned iter budget. Default {DEFAULT_ITER_MIN}.')
    parser.add_argument('--iter-max', type=int, default=DEFAULT_ITER_MAX,
                        help=f'Ceiling on the retuned iter budget. Default {DEFAULT_ITER_MAX}.')
    parser.add_argument('--descriptiveness-threshold', type=float,
                        default=DEFAULT_DESCRIPTIVENESS_THRESHOLD,
                        help=f'Starting ratePrompt threshold (0-100). '
                             f'Default {DEFAULT_DESCRIPTIVENESS_THRESHOLD}.')
    parser.add_argument('--quality-target', type=float, default=DEFAULT_QUALITY_TARGET,
                        help=f'Target quality-signal value for the retune. '
                             f'Default {DEFAULT_QUALITY_TARGET}. Note that the '
                             f'natural range depends on --quality-signal: '
                             f'CLIP cosines are typically 0.20-0.40; structured-feature '
                             f'agreement is in [0, 1].')
    parser.add_argument('--quality-signal', choices=['clip_text', 'structured_features'],
                        default='clip_text',
                        help='Quality signal for the agent loop. clip_text: CLIP '
                             'image-vs-text cosine. structured_features: per-field '
                             'agreement against a reference 13-field feature dict '
                             'extracted from --reference. Default: clip_text.')
    parser.add_argument('--reference', choices=['title', 'initial_prompt'],
                        default='title',
                        help='Reference text/features that the quality signal '
                             'compares against. title: product title from '
                             'metadata.json (v1). initial_prompt: contents of '
                             'initial_prompt.txt (v2) or its pre-extracted '
                             'structured_features_initial_v1.json (v3). '
                             'Default: title.')
    args = parser.parse_args()

    # Validate (quality_signal, reference) combination early.
    valid_combos = {
        ('clip_text', 'title'),
        ('clip_text', 'initial_prompt'),
        ('structured_features', 'initial_prompt'),
    }
    if (args.quality_signal, args.reference) not in valid_combos:
        parser.error(
            f'unsupported (--quality-signal, --reference) combination: '
            f'({args.quality_signal!r}, {args.reference!r}). '
            f'Valid combinations: {sorted(valid_combos)}'
        )

    load_dotenv()

    products = discover_products()
    if not products:
        print(f'[!] No products found under {DATA_DIR}/ with all of '
              f'(metadata.json, initial_prompt.txt, reviews_ranked.jsonl).')
        sys.exit(1)

    selected = resolve_only(products, args.only) if args.only else products
    models_to_run = resolve_models(args.model)

    print(f'Config name: {args.config_name!r}  '
          f'(promote={"no" if args.no_promote else "yes"})')
    print(f'Quality signal: {args.quality_signal}  |  Reference: {args.reference}')
    print(f'Discovered products: {sorted(products)}')
    print(f'Selected: {sorted(selected)}')
    print(f'Models to run: {[MODEL_TAGS[m] for m in models_to_run]}')
    print(f'Hyperparameters: image_count={args.image_count}  '
          f'iter_start={args.iter_start}  iter_min={args.iter_min}  iter_max={args.iter_max}  '
          f'descriptiveness_threshold={args.descriptiveness_threshold}  '
          f'quality_target={args.quality_target}')
    print()

    # Load Mistral ONCE. This is the expensive bit (~30s + ~5 GB VRAM). It is
    # reused across every (product, model) pair in this run.
    print('Loading Mistral...')
    model, tokenizer = load_mistral()
    print()

    results = []
    for slug, pdir in selected.items():
        # One dataloader per product, shared across both model runs so both
        # model trajectories see identical review batches.
        dataloader = make_reviews_dataloader(slug)

        for image_model_num in models_to_run:
            r = run_one(slug, pdir, image_model_num, dataloader, model, tokenizer, args)
            results.append(r)

        # Canonical `converged_prompt.txt` is written from the FLUX run
        # inside run_one (tag == 'flux' branch) when --no-promote is not set.
        # If only model 2 was run this session AND we want a canonical
        # global prompt, fall back to the GPT per-config file. This only
        # applies when both: promotion is on AND we didn't run FLUX.
        if not args.no_promote and 1 not in models_to_run:
            pc_gpt = per_config_paths(pdir, 'gpt', args.config_name)
            can = canonical_paths(pdir, 'gpt')
            if os.path.exists(pc_gpt['converged']):
                shutil.copyfile(pc_gpt['converged'], can['converged_global'])

    # Summary.
    print()
    print('=' * 72)
    print('  DONE')
    print('=' * 72)
    ok = [r for r in results if not r.get('skipped') and not r.get('error')]
    skipped = [r for r in results if r.get('skipped')]
    errors = [r for r in results if r.get('error')]
    for r in ok:
        q_best = r['qualities'][r['best_idx']]
        promo = 'promoted' if r.get('promoted') else 'not-promoted'
        print(f'  {r["slug"]:<18} {r["model_tag"]:<5} '
              f'best_q={q_best:.4f}  best_idx={r["best_idx"]}  '
              f'{r["elapsed_seconds"]}s  ({promo})')
    if skipped:
        print(f'  skipped: {[(r["slug"], r["model_tag"], r["config_name"]) for r in skipped]}')
    if errors:
        print(f'  errors:  {[(r["slug"], r["model_tag"], r["config_name"], r["error"]) for r in errors]}')


if __name__ == '__main__':
    main()
