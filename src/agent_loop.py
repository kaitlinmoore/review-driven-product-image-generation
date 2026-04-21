'''agent_loop.py
-------------
Outer-loop orchestration for iterative image generation.

Structure:
- Quadratic-fit helpers: fit_quadratic, solve_for_x, pick_valid_solution,
  cumulative_iters. Used to retune descriptiveness threshold and iteration
  count across image generations based on observed (x, quality) curves.
- genImage(prompt, imageModelNum): thin dispatcher that routes to one of the
  two wired wrappers (FLUX.1-schnell as model 1, gpt-image-1.5 as model 2).
  This replaces the notebook's local SD 1.5 implementation.
- agentLoop: runs PromptWriter.improvementLoop + genImage + evalImage over
  imageCount iterations. Per-iteration artifacts (generated PNG + metadata
  JSON) are saved under runs/run_{timestamp}/iteration_{i}/.

Per-run outputs land in runs/run_{YYYYMMDD_HHMMSS}/ (relative to cwd, i.e.
the repo root when invoked via `python src/agent_loop.py` or similar).
'''

import json
import math
import os
from datetime import datetime
from typing import Callable

import torch
from PIL import Image as PILImage

from prompt_writer import PromptWriter
from gen_image_flux import generate_flux
from gen_image_gpt import generate_gpt_image


## CONSTANTS ##

DEFAULT_RUN_ROOT = 'runs'
DEFAULT_QUALITY_TARGET = 0.5
DEFAULT_IMAGE_COUNT = 2
DEFAULT_IMAGE_MODEL_NUM = 1   # 1 = FLUX.1-schnell, 2 = gpt-image-1.5


## QUADRATIC FIT HELPERS ##

def fit_quadratic(x: torch.Tensor, y: torch.Tensor):
    '''Fit y ≈ a*x^2 + b*x + c by least squares. Returns (a, b, c) tensors.'''
    x = x.float()
    y = y.float()
    
    # Design matrix [x^2, x, 1]
    A = torch.stack([x**2, x, torch.ones_like(x)], dim=1)
    coeffs = torch.linalg.lstsq(A, y).solution  # shape (3,)
    a, b, c = coeffs
    return a, b, c


def solve_for_x(a, b, c, k):
    '''Solve a*x^2 + b*x + c = k. Returns (x1, x2) or None if no real roots.'''
    discriminant = b**2 - 4 * a * (c - k)
    if discriminant < 0:
        return None
    sqrt_disc = torch.sqrt(discriminant)
    x1 = (-b + sqrt_disc) / (2 * a)
    x2 = (-b - sqrt_disc) / (2 * a)
    return x1, x2


def cumulative_iters(iters):
    '''Cumulative sum of a per-image iteration-count list.'''
    return torch.cumsum(torch.tensor(iters, dtype=torch.float32), dim=0)


def pick_valid_solution(solutions, current_value):
    '''From a pair of quadratic roots, pick the closest non-negative root to
    current_value. Returns None if neither is valid.'''
    if solutions is None:
        return None
    x1, x2 = solutions
    candidates = torch.tensor([x1, x2])
    candidates = candidates[~torch.isnan(candidates)]
    candidates = candidates[candidates >= 0]
    if len(candidates) == 0:
        return None
    return candidates[torch.argmin(torch.abs(candidates - current_value))]


## IMAGE DISPATCH ##

def genImage(prompt: str, imageModelNum: int = DEFAULT_IMAGE_MODEL_NUM):
    '''Dispatch to one of the two image-generation wrappers.

    1 = FLUX.1-schnell (open-weights, routed via HuggingFace InferenceClient
        to the fal-ai provider).
    2 = gpt-image-1.5 (proprietary, via the OpenAI API).

    Returns PIL.Image in RGB mode.'''
    if imageModelNum == 1:
        return generate_flux(prompt)
    if imageModelNum == 2:
        return generate_gpt_image(prompt)
    raise ValueError(f'imageModelNum must be 1 or 2, got {imageModelNum!r}')


## AGENT LOOP ##

def agentLoop(
    dataloader,
    model,
    tokenizer,
    quality_signal_fn: Callable[[PILImage.Image], float],
    descriptivenessThreshold: float,
    iterStart: int,
    iterMin: int,
    iterMax: int,
    imageCount: int = DEFAULT_IMAGE_COUNT,
    imageModelNum: int = DEFAULT_IMAGE_MODEL_NUM,
    qualityTarget: float = DEFAULT_QUALITY_TARGET,
    run_root: str = DEFAULT_RUN_ROOT,
    initial_prompt_path: str | None = None,
):
    '''Outer orchestration: generate imageCount images, refitting descriptiveness
    threshold and per-image iteration budget via quadratic fit between runs.

    quality_signal_fn(img: PIL.Image) -> float in [0, 1] (or thereabouts).
    Higher = better. Driver constructs this closure based on --quality-signal
    and --reference flags; agentLoop is signal-agnostic. Examples:
    - lambda img: compImage(img, title)                       # v1
    - lambda img: compImage(img, initial_prompt_text)         # v2
    - lambda img: eval_structured_features(img, ref_features) # v3

    Returns (itersTaken, descriptivenesses, prompts, images, qualities, run_dir).
    The first five are lists of length imageCount; run_dir is the absolute-ish
    path (relative to cwd) of the timestamped run output directory under
    run_root, so callers can locate per-iteration artifacts after return.'''
    # Create run directory under run_root (default: ./runs/).
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(run_root, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    promptWriter = PromptWriter(dataloader, model, tokenizer, initial_prompt_path=initial_prompt_path)
    itersTaken, descriptivenesses, prompts, images, qualities = [], [], [], [], []

    for i in range(imageCount):
        iter_dir = os.path.join(run_dir, f'iteration_{i}')
        os.makedirs(iter_dir, exist_ok=True)

        # 1. Refine prompt until descriptiveness hits threshold OR iterStart cap.
        current_iters = promptWriter.improvementLoop(descriptivenessThreshold, iterStart)
        itersTaken.append(current_iters)

        current_desc = promptWriter.descriptiveness
        descriptivenesses.append(current_desc)

        current_prompt = promptWriter.prompt
        prompts.append(current_prompt)

        # 2. Generate image.
        img = genImage(current_prompt, imageModelNum)
        images.append(img)

        # 3. Evaluate via the driver-supplied quality_signal_fn closure.
        # The function signature is opaque to agentLoop — the driver decided
        # which metric (CLIP image-vs-text, structured-feature agreement, etc.)
        # and which reference (title, initial_prompt, ...) to use.
        q_score = quality_signal_fn(img)
        qualities.append(q_score)

        # Save per-iteration artifacts.
        img.save(os.path.join(iter_dir, 'generated_image.png'))
        with open(os.path.join(iter_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'prompt': current_prompt,
                'score': q_score,
                'iters_taken': current_iters,
                'descriptiveness': current_desc,
            }, f, indent=2, ensure_ascii=False)

        # Retune descriptivenessThreshold and iterStart for the NEXT image
        # via quadratic fits against observed quality scores.
        zero = torch.tensor([0.0], dtype=torch.float32)
        descriptivenesses_t = torch.cat(
            [zero, torch.tensor(descriptivenesses, dtype=torch.float32)])
        qualities_t = torch.cat(
            [zero, torch.tensor(qualities, dtype=torch.float32)])
        iter_cumsum_t = torch.cat(
            [zero, torch.cumsum(torch.tensor(itersTaken, dtype=torch.float32), dim=0)])

        a_d, b_d, c_d = fit_quadratic(descriptivenesses_t, qualities_t)
        a_i, b_i, c_i = fit_quadratic(iter_cumsum_t, qualities_t)

        desc_sol = pick_valid_solution(
            solve_for_x(a_d, b_d, c_d, torch.tensor(qualityTarget)),
            descriptivenesses_t[-1],
        )
        iter_sol = pick_valid_solution(
            solve_for_x(a_i, b_i, c_i, torch.tensor(qualityTarget)),
            iter_cumsum_t[-1],
        )

        if desc_sol is not None:
            descriptivenessThreshold = desc_sol.item()
        if iter_sol is not None:
            iterStart = min(
                max(math.ceil(iter_sol.item() - iter_cumsum_t[-1].item()), iterMin),
                iterMax,
            )

    return itersTaken, descriptivenesses, prompts, images, qualities, run_dir
