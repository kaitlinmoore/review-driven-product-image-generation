'''gen_image_flux.py
-----------------
Image generation via FLUX.1-schnell on the HuggingFace Inference router,
routing to the fal-ai provider by default. One of two image-generation
models used for the Q3 comparison.

Returns PIL.Image in RGB mode for downstream compImage / evaluation / display.
Uses huggingface_hub.InferenceClient rather than raw HTTP because HuggingFace
deprecated the old api-inference.huggingface.co/models/... URL pattern in
2024-2025. The router approach is the currently supported path and also
survives future HF infrastructure changes.

Auth: reuses HUGGINGFACE_TOKEN (canonical), HF_TOKEN (shorthand), or
HUGGING_FACE_TOKEN (non-canonical) in that order of preference. No separate
fal.ai / Replicate account needed; HF bills and settles with the provider.

Usage and CLI FLags:
    from gen_image_flux import generate_flux
    img = generate_flux(prompt)
    img.save('out.png')

Notes:
- FLUX.1-schnell is distilled for 1-4 step inference; DEFAULT_NUM_STEPS=4 is
  the HF model card's recommended sweet spot.
- Provider default is fal-ai (fastest for FLUX in practice, explicitly
  listed on the model card). Swap to 'replicate' or 'together' via the
  `provider` arg if fal has an outage.
'''

import os
import sys
import time

from PIL import Image
# Use for locally .env defined API keys/tokens.
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from replay import cached_call


## CONSTANTS ##

FLUX_MODEL = 'black-forest-labs/FLUX.1-schnell'

DEFAULT_PROVIDER = 'fal-ai'    # also 'replicate', 'together', 'nebius', ...
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_NUM_STEPS = 4          # schnell is distilled; 1-4 steps is the sweet spot
DEFAULT_MAX_RETRIES = 2

# Env-var names we'll accept for the HF token, in preference order.
HF_TOKEN_ENV_NAMES = ('HUGGINGFACE_TOKEN', 'HF_TOKEN', 'HUGGING_FACE_TOKEN')


## HELPERS ##

def _token() -> str:
    '''Read HF token from .env. Accepts three naming variants so an .env with
    HUGGING_FACE_TOKEN (non-canonical) or HF_TOKEN (shorthand) still works.'''
    load_dotenv()
    for name in HF_TOKEN_ENV_NAMES:
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip()
    raise RuntimeError(
        f'No HuggingFace token found. Set one of {list(HF_TOKEN_ENV_NAMES)} in .env.'
    )


def _client(provider: str) -> InferenceClient:
    return InferenceClient(provider=provider, api_key=_token())


## PUBLIC API ##

def generate_flux(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_inference_steps: int = DEFAULT_NUM_STEPS,
    provider: str = DEFAULT_PROVIDER,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Image.Image:
    '''Generate one image via FLUX.1-schnell. Returns PIL.Image in RGB mode.
    Retries transient failures with exponential backoff.

    Replay-aware: in record mode, a cache hit skips the live call; in replay
    mode, a missing cache raises FileNotFoundError. Cache key is computed
    from the prompt + geometry + provider (no seed, so repeat calls with
    identical inputs all return the first-observed cached image).'''
    inputs = {
        'model': FLUX_MODEL,
        'prompt': prompt,
        'width': width,
        'height': height,
        'num_inference_steps': num_inference_steps,
        'provider': provider,
    }

    def _live() -> Image.Image:
        client = _client(provider)
        last_err: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                img = client.text_to_image(
                    prompt,
                    model=FLUX_MODEL,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                )
                return img.convert('RGB')
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    print(f'[FLUX/{provider}] {type(e).__name__}, retrying in {wait}s '
                          f'(attempt {attempt + 1}/{max_retries})')
                    time.sleep(wait)
                    continue
                break

        raise RuntimeError(
            f'FLUX generation failed via provider={provider!r} after '
            f'{max_retries + 1} attempts: {last_err}'
        )

    return cached_call('flux_gen', inputs, _live, format='png')


## SMOKE TEST ##

if __name__ == '__main__':
    prompt = (
        sys.argv[1] if len(sys.argv) > 1
        else 'a wooden chess set on a dark mahogany table, studio lighting'
    )
    print(f'Generating via {DEFAULT_PROVIDER}: {prompt!r}')
    t0 = time.time()
    img = generate_flux(prompt)
    elapsed = time.time() - t0
    out_path = 'flux_smoke_test.png'
    img.save(out_path)
    print(f'Saved {img.size[0]}x{img.size[1]} to {out_path} in {elapsed:.1f}s')
