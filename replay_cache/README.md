# replay_cache/

Cached outputs of external API calls and heavy model inference, keyed by a
deterministic hash of the call's function name and inputs. Lets the pipeline
be re-run offline against the canonical record, without API keys or GPU.

## Layout

```
replay_cache/
├── README.md                          # this file
├── .gitkeep                           # keeps the directory tracked when empty
├── {fn_name}_{key}.json               # JSON-encoded outputs (default)
├── {fn_name}_{key}.txt                # plain-text outputs (e.g. step_prompt)
├── {fn_name}_{key}.png                # PIL.Image outputs (FLUX, gpt-image)
└── {fn_name}_{key}.bin                # raw bytes outputs (rarely used)
```

All entries are flat files — no subdirectories per call.

- `{fn_name}` is the caller tag (e.g. `initial_prompt`, `step_prompt`,
  `flux_gen`, `eval_clip_image`).
- `{key}` is the first 16 hex chars of
  `SHA256(json({"fn": fn_name, "inputs": inputs}, sort_keys=True))`.

For the full inventory of currently-used `fn_name` values, see the
"What Gets Cached" table in `REPRODUCIBILITY.md`.

## Wrapper API

`src/replay.py` exports `cached_call` and `path_hash`. Typical usage:

```python
from replay import cached_call, path_hash

def _live():
    # body that calls the live API / model
    return result

result = cached_call(
    fn_name='initial_prompt',
    inputs={                           # everything that affects the output
        'model': 'gpt-4o',
        'temperature': 0,
        'prompt_version': 'v1',
        'prompt_sha256': '...',        # hash the system prompt so edits bust cache
        'context_text': '...',
    },
    live_fn=_live,
    format='json',                     # 'json' | 'text' | 'png' | 'bytes'
    mode=None,                         # None → reads REPLAY_MODE env var
)
```

When an input is a local file path, wrap it with `path_hash(path)` so the
cache key is stable across machines with different absolute paths. When an
input is a `PIL.Image` (as in the image-vs-image evaluation calls), use
`_image_hash` from `eval_image.py` to hash pixel bytes.

## Modes

Selected by the `REPLAY_MODE` environment variable (or the per-call `mode=`
override):

- `record` (default) — cache hit returns cached value; cache miss calls
  `live_fn`, saves the result, and returns it. Safe memoization.
- `replay` — cache hit returns cached value; cache miss raises
  `FileNotFoundError`. Grader path — no live calls ever.
- `force_record` — always calls `live_fn`, overwrites the cache entry,
  returns. Used when intentionally re-recording a call after a prompt or
  model change.

## What's committed

The canonical record's cache is committed to git so a grader can clone and
run in replay mode with zero API setup. That's the whole point — excluding
cache contents would defeat the purpose. The `.gitignore` at the repo root
intentionally does not exclude anything under `replay_cache/`.
