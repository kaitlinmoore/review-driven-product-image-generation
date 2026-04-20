'''replay.py
---------
Replay-cache wrapper. Caches external API call results to disk so the full
pipeline can be re-run deterministically without live API / GPU access.

Modes (selected via the REPLAY_MODE environment variable, or per-call `mode`):
- 'record' (default): if cache exists → return cached; else → call live,
  save, return. Safe memoization, never invalidates an existing cache entry.
- 'replay': if cache exists → return cached; else → raise FileNotFoundError.
  Strict reproduction mode. Graders run this with no API access.
- 'force_record': always call live, overwrite cache, return. Use to
  re-record specific calls when iterating.

Output formats supported: 'json' (default), 'text', 'png', 'bytes'.

Cache file naming: replay_cache/{fn_name}_{key}.{ext}
where key = SHA256(json({'fn': fn_name, 'inputs': inputs}, sort_keys=True))[:16].

Basic usage:
    from replay import cached_call, path_hash

    # Text-producing call (default format='json')
    def _live_filter():
        return {'passes_gate': True, 'reason': 'describes material and shape'}

    result = cached_call(
        fn_name='filter_gate',
        inputs={'review_id': 'abc_123', 'prompt_version': 'v1'},
        live_fn=_live_filter,
        format='json',
    )

    # Image-producing call (format='png' returns PIL.Image)
    from gen_image_flux import generate_flux
    img = cached_call(
        fn_name='flux_gen',
        inputs={'prompt': 'a chess set', 'model': 'FLUX.1-schnell', 'seed': None},
        live_fn=lambda: generate_flux('a chess set'),
        format='png',
    )

    # When an input IS a file path, hash its contents so the key is stable
    # across machines with different absolute paths:
    inputs = {
        'prompt': '...',
        'image_hash': path_hash('data/chess_set/images/main.jpg'),
    }

Integration pattern (opt-in per wrapper):
    Each wrapper that calls a live API stays unchanged by default; callers
    opt into caching by calling cached_call around the live call. To wrap
    an entire wrapper, the typical 3-step change is:
        1. Wrap the existing body into a local _live() function
        2. Build an inputs dict of everything that affects the result
        3. Call cached_call(fn_name, inputs, _live, format=...)

'''

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Literal

from PIL import Image


## CONSTANTS ##

REPLAY_DIR = Path('replay_cache')

MODE_RECORD = 'record'
MODE_REPLAY = 'replay'
MODE_FORCE_RECORD = 'force_record'
VALID_MODES = (MODE_RECORD, MODE_REPLAY, MODE_FORCE_RECORD)

REPLAY_MODE = os.environ.get('REPLAY_MODE', MODE_RECORD)
if REPLAY_MODE not in VALID_MODES:
    raise ValueError(
        f'REPLAY_MODE={REPLAY_MODE!r} invalid. Must be one of {VALID_MODES}.'
    )

FormatName = Literal['json', 'text', 'png', 'bytes']

_EXTENSIONS: dict[str, str] = {
    'json': '.json',
    'text': '.txt',
    'png': '.png',
    'bytes': '.bin',
}


## HASHING ##

def _cache_key(fn_name: str, inputs: dict) -> str:
    '''Deterministic 16-char hex key from (fn_name, inputs).
    inputs must be JSON-serializable; sort_keys=True guarantees order independence.'''
    payload = json.dumps({'fn': fn_name, 'inputs': inputs}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def path_hash(path: str | Path) -> str:
    '''16-char hex hash of a file's contents. Use in an `inputs` dict rather
    than the raw path string so cache keys are stable across machines with
    different absolute paths to the same bytes.'''
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'path_hash: file not found: {p}')
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


## FORMAT DISPATCH ##

def _cache_path(fn_name: str, key: str, format: FormatName) -> Path:
    if format not in _EXTENSIONS:
        raise ValueError(f'Unsupported format {format!r}. Use one of {list(_EXTENSIONS)}.')
    return REPLAY_DIR / f'{fn_name}_{key}{_EXTENSIONS[format]}'


def _save(path: Path, result: Any, format: FormatName) -> None:
    '''Atomic write: writes to path+.tmp then os.replace to final path.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + '.tmp')

    if format == 'json':
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, sort_keys=True)
    elif format == 'text':
        if not isinstance(result, str):
            raise TypeError(f"format='text' expects str, got {type(result).__name__}")
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(result)
    elif format == 'png':
        if not isinstance(result, Image.Image):
            raise TypeError(f"format='png' expects PIL.Image, got {type(result).__name__}")
        result.save(tmp, format='PNG')
    elif format == 'bytes':
        if not isinstance(result, (bytes, bytearray)):
            raise TypeError(f"format='bytes' expects bytes, got {type(result).__name__}")
        with open(tmp, 'wb') as f:
            f.write(result)
    else:
        raise ValueError(f'Unsupported format: {format!r}')

    os.replace(tmp, path)


def _load(path: Path, format: FormatName) -> Any:
    '''Read and decode a cached file according to its format.'''
    if format == 'json':
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    if format == 'text':
        return path.read_text(encoding='utf-8')
    if format == 'png':
        return Image.open(path).convert('RGB')
    if format == 'bytes':
        return path.read_bytes()
    raise ValueError(f'Unsupported format: {format!r}')


## PUBLIC API ##

def cached_call(
    fn_name: str,
    inputs: dict,
    live_fn: Callable[[], Any],
    format: FormatName = 'json',
    mode: str | None = None,
) -> Any:
    '''Cache-aware wrapper for an external API or other expensive call.

    Args:
        fn_name: short identifier for the kind of call (e.g. 'filter_gate',
            'flux_gen', 'step_prompt'). Used as the cache filename prefix.
        inputs: dict of everything that deterministically affects the result.
            Must be JSON-serializable. Hashed to form the cache key.
        live_fn: zero-arg callable that performs the real API call.
            Only invoked on cache miss (or in force_record mode).
        format: one of 'json', 'text', 'png', 'bytes'. Default 'json'.
        mode: override the module-level REPLAY_MODE per call. If None,
            uses the env var REPLAY_MODE (default 'record').

    Returns:
        The cached (or freshly generated) result, typed by `format`:
        dict/list/etc for 'json', str for 'text', PIL.Image for 'png',
        bytes for 'bytes'.

    Raises:
        FileNotFoundError: in 'replay' mode when the cache entry is missing.
        ValueError: if mode or format is invalid.
        TypeError: if the result type doesn't match the declared format.
    '''
    active_mode = mode or REPLAY_MODE
    if active_mode not in VALID_MODES:
        raise ValueError(f'mode={active_mode!r} invalid. Must be one of {VALID_MODES}.')

    key = _cache_key(fn_name, inputs)
    path = _cache_path(fn_name, key, format)

    # Replay: strict. Never call live.
    if active_mode == MODE_REPLAY:
        if not path.exists():
            raise FileNotFoundError(
                f'No replay data for {fn_name} (key={key}) at {path}. '
                f'Re-run in record mode first, or check that inputs match '
                f'the original record-mode run.'
            )
        return _load(path, format)

    # Record (default): memoize | return cached if exists, else call live.
    if active_mode == MODE_RECORD and path.exists():
        return _load(path, format)

    # Record with no cache hit, OR force_record: call live and save.
    result = live_fn()
    _save(path, result, format)
    return result


def list_cache(fn_name: str | None = None) -> list[Path]:
    '''List all cache files, optionally filtered to a specific fn_name.
    Useful for debugging and for computing "how many calls have been cached."'''
    if not REPLAY_DIR.exists():
        return []
    valid_suffixes = set(_EXTENSIONS.values())
    if fn_name is None:
        return sorted(p for p in REPLAY_DIR.iterdir()
                      if p.is_file() and p.suffix in valid_suffixes)
    prefix = f'{fn_name}_'
    return sorted(p for p in REPLAY_DIR.iterdir()
                  if p.is_file()
                  and p.name.startswith(prefix)
                  and p.suffix in valid_suffixes)


## SELF-TEST ##

if __name__ == '__main__':
    # Round-trip test for all formats and modes. No external deps, no model
    # downloads. Leaves replay_cache/ in the state it started (selftest
    # artifacts cleaned up in finally).
    import sys
    import tempfile

    test_slugs_created: list[Path] = []

    def _cleanup():
        for p in test_slugs_created:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    try:
        # 1. JSON round-trip and memoization
        calls_made = {'n': 0}

        def _live_json():
            calls_made['n'] += 1
            return {'key': 'value', 'items': [1, 2, 3]}

        r1 = cached_call('selftest_json', {'x': 1}, _live_json,
                         format='json', mode='force_record')
        r2 = cached_call('selftest_json', {'x': 1}, _live_json,
                         format='json', mode='record')
        test_slugs_created.append(_cache_path('selftest_json', _cache_key('selftest_json', {'x': 1}), 'json'))
        assert r1 == r2 == {'key': 'value', 'items': [1, 2, 3]}
        assert calls_made['n'] == 1, (
            f"record mode should hit cache after force_record; live_fn was called {calls_made['n']}x"
        )

        # 2. Replay mode raises on missing key.
        try:
            cached_call('selftest_nope', {'x': 1}, _live_json, mode='replay')
            print('FAIL: replay mode should have raised on missing cache')
            sys.exit(1)
        except FileNotFoundError:
            pass

        # 3. Replay mode returns cached value when present.
        r3 = cached_call('selftest_json', {'x': 1}, _live_json, mode='replay')
        assert r3 == r1
        assert calls_made['n'] == 1  # still no new live calls

        # 4. force_record always calls live, overwriting cache.
        r4 = cached_call('selftest_json', {'x': 1}, _live_json,
                         format='json', mode='force_record')
        assert calls_made['n'] == 2

        # 5. Text round-trip
        def _live_text():
            return 'hello world\nline 2\n'
        t_rec = cached_call('selftest_text', {'y': 2}, _live_text,
                            format='text', mode='force_record')
        t_rep = cached_call('selftest_text', {'y': 2}, _live_text,
                            format='text', mode='replay')
        test_slugs_created.append(_cache_path('selftest_text', _cache_key('selftest_text', {'y': 2}), 'text'))
        assert t_rec == t_rep == 'hello world\nline 2\n'

        # 6. PNG round-trip ( tiny 8x8 orange square )
        img_in = Image.new('RGB', (8, 8), color=(255, 128, 0))
        p_rec = cached_call('selftest_png', {'z': 3}, lambda: img_in,
                            format='png', mode='force_record')
        p_rep = cached_call('selftest_png', {'z': 3}, lambda: img_in,
                            format='png', mode='replay')
        test_slugs_created.append(_cache_path('selftest_png', _cache_key('selftest_png', {'z': 3}), 'png'))
        assert p_rec.size == p_rep.size == (8, 8)
        assert p_rep.mode == 'RGB'
        assert p_rep.getpixel((0, 0)) == (255, 128, 0)

        # 7. Bytes round-trip
        data_in = b'\x00\x01\x02\xff' * 32
        b_rec = cached_call('selftest_bytes', {'w': 4}, lambda: data_in,
                            format='bytes', mode='force_record')
        b_rep = cached_call('selftest_bytes', {'w': 4}, lambda: data_in,
                            format='bytes', mode='replay')
        test_slugs_created.append(_cache_path('selftest_bytes', _cache_key('selftest_bytes', {'w': 4}), 'bytes'))
        assert b_rec == b_rep == data_in

        # 8. path_hash is stable and length=16.
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b'deterministic content for hashing')
            tmp_name = f.name
        try:
            h1 = path_hash(tmp_name)
            h2 = path_hash(tmp_name)
            assert h1 == h2
            assert len(h1) == 16
        finally:
            os.unlink(tmp_name)

        # 9. Different inputs yield different keys.
        k_a = _cache_key('f', {'x': 1})
        k_b = _cache_key('f', {'x': 2})
        k_c = _cache_key('g', {'x': 1})
        assert k_a != k_b, 'different input values must yield different keys'
        assert k_a != k_c, 'different fn_name must yield different key'

        # 10. Dict order doesn't affect key (sort_keys=True in serialization).
        k_d = _cache_key('f', {'a': 1, 'b': 2})
        k_e = _cache_key('f', {'b': 2, 'a': 1})
        assert k_d == k_e, 'dict key order must not affect cache key'

        # 11. Type-mismatch errors
        try:
            cached_call('selftest_typemismatch', {'x': 0}, lambda: 123,
                        format='text', mode='force_record')
            print('FAIL: should have raised TypeError for text format + int result')
            sys.exit(1)
        except TypeError:
            pass

        # 12. list_cache finds our selftest files.
        selftest_files = list_cache()
        selftest_matching = [p for p in selftest_files if p.name.startswith('selftest_')]
        assert len(selftest_matching) >= 4, (
            f'expected >=4 selftest files, found {len(selftest_matching)}'
        )
        scoped = list_cache('selftest_png')
        assert len(scoped) == 1 and scoped[0].name.startswith('selftest_png_')

        print('All self-test checks passed.')
        print(f'REPLAY_DIR  = {REPLAY_DIR.resolve()}')
        print(f'REPLAY_MODE = {REPLAY_MODE!r}')
        print(f'Formats supported: {list(_EXTENSIONS)}')

    finally:
        _cleanup()
