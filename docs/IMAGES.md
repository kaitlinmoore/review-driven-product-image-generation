# Where to Find the Generated Images

Each image and every per-iteration intermediate lives in the
`data/` directory of the submitted package, in a flat naming convention.
This document is the lookup reference: given a (product, model, config)
combination, where the corresponding image, prompt, and metadata files live on disk.

## Filepath Patterns

| Artifact | Path |
|---|---|
| Generated image (canonical, best iteration) | `data/{product}/generated_image_{model}_{config}.png` |
| Refined prompt that produced the canonical image | `data/{product}/converged_prompt_{model}_{config}.txt` |
| Per-iteration generated image (all 3 attempts) | `data/{product}/trajectories/{model}_{config}/iteration_{N}/generated_image.png` |
| Per-iteration prompt | `data/{product}/trajectories/{model}_{config}/iteration_{N}/prompt.txt` |
| Per-iteration metadata (quality, descriptiveness, iters_taken) | `data/{product}/trajectories/{model}_{config}/iteration_{N}/metadata.json` |
| Run metadata (which iteration was selected as canonical, etc.) | `data/{product}/agent_run_{model}_{config}_meta.json` |
| Ground-truth product photos | `data/{product}/images/main.jpg`, `alt_01.jpg`, … |

## Legal Values

- `{product}`: `backpack`, `chess_set`, `espresso_machine`, `headphones`, `jeans`, `water_bottle`
- `{model}`: `flux` (FLUX.1-schnell), `gpt` (gpt-image-1.5)
- `{config}`: `v1_title_clip`, `v2_initial_prompt_clip`, `v3_initial_prompt_features`
- `{N}`: `0`, `1`, `2`

This yields **36 canonical generated images** (6 products × 2 models × 3
configs) and **108 per-iteration generated images** (36 runs × 3 attempts
each).

## Worked Examples

For each product, the FLUX v3 image, the gpt-image v3 image, and the
ground-truth main reference photo:

| Product | FLUX.1-schnell (v3) | gpt-image-1.5 (v3) | Ground truth |
|---|---|---|---|
| chess_set | `data/chess_set/generated_image_flux_v3_initial_prompt_features.png` | `data/chess_set/generated_image_gpt_v3_initial_prompt_features.png` | `data/chess_set/images/main.jpg` |
| water_bottle | `data/water_bottle/generated_image_flux_v3_initial_prompt_features.png` | `data/water_bottle/generated_image_gpt_v3_initial_prompt_features.png` | `data/water_bottle/images/main.jpg` |
| jeans | `data/jeans/generated_image_flux_v3_initial_prompt_features.png` | `data/jeans/generated_image_gpt_v3_initial_prompt_features.png` | `data/jeans/images/main.jpg` |
| backpack | `data/backpack/generated_image_flux_v3_initial_prompt_features.png` | `data/backpack/generated_image_gpt_v3_initial_prompt_features.png` | `data/backpack/images/main.jpg` |

To compare the same product under a different quality-signal config,
replace `v3_initial_prompt_features` with `v1_title_clip` or
`v2_initial_prompt_clip`.

## Inspecting the Refinement Trajectory

To see how the agent loop's prompt and image evolved across the three
image attempts for a given run, browse the `trajectories/` subdirectory.
Example:

FLUX v3 chess-set run:

```
data/chess_set/trajectories/flux_v3_initial_prompt_features/
    iteration_0/
        prompt.txt           ← refined prompt used for this attempt
        generated_image.png  ← image produced from that prompt
        metadata.json        ← quality score, descriptiveness, iters_taken
    iteration_1/  …
    iteration_2/  …
```

The canonical image at the top level
(`generated_image_flux_v3_initial_prompt_features.png`) is a copy of
whichever iteration scored highest under that config's quality signal. The
selected iteration index is recorded as `best_idx` in
`data/{product}/agent_run_{model}_{config}_meta.json`, alongside the
per-iteration `qualities` array. For the FLUX v3 chess-set run,
`best_idx = 1` and `qualities = [0.412, 0.429, 0.269]` — the agent's
second attempt scored highest and was canonicalised.

## Replay Reproduction

Running the pipeline with `REPLAY_MODE=replay` (see `REPRODUCIBILITY.md`)
regenerates every file listed above from the committed `replay_cache/`,
deterministically, with no API costs and no GPU. The on-disk layout after
replay is identical to the layout in the submitted package.
