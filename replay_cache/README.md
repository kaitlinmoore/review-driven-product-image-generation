# replay_cache/

Cached outputs of external API calls, keyed by a deterministic hash of the
call's function name and inputs. Lets the pipeline be re-run offline against
the canonical record, without API keys or GPU.

## Intended Layout

```
replay_cache/
├── README.md                          # this file
├── .gitkeep                           # keeps the directory tracked when empty
├── {fn_name}_{key}.json               # text-producing calls
└── {fn_name}_{key}/                   # binary-producing calls
    └── image.png
```

`{fn_name}` is the caller (e.g. `initial_prompt`, `step_prompt`,
`rate_prompt`, `gen_image`). `{key}` is the first 16 hex chars of
`SHA256(json({"fn": fn_name, "inputs": inputs}))`.

## Design Source

The wrapper that populates this directory is defined in replay.py.

```python
REPLAY_DIR = Path('replay_cache')
REPLAY_MODE = os.environ.get("REPLAY_MODE", "record")  # 'record' or 'replay'

def cached_call(fn_name, inputs, live_fn, output_encoder, output_decoder):
    key = _cache_key(fn_name, inputs)
    cache_path = REPLAY_DIR / f'{fn_name}_{key}.json'
    if REPLAY_MODE == 'replay':
        if not cache_path.exists():
            raise FileNotFoundError(f'No replay data for {fn_name} key {key}')
        return output_decoder(json.loads(cache_path.read_text()))
    # record: call live_fn, save, return
    result = live_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(output_encoder(result)))
    return result
```

## Status

The directory is scaffolded so the wrapper can drop into `src/replay.py`
without any layout changes downstream. Outstanding work (from the KMM TODO
list in `working_image_refinement.ipynb` cell 0):

- Confirm append semantics, not overwrite.
- Handle image outputs (`.png` files) alongside JSON.
- Allow inputs that reference local file paths (e.g. prompt text file).
- Wrap all external API calls in the pipeline:

Once wired, setting `REPLAY_MODE=replay` will let a grader reproduce the
canonical run without API keys, gated-model access, or a GPU.
