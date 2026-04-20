'''gen_image_gpt.py
----------------
Image generation via OpenAI's gpt-image-1.5 (the successor to DALL-E 3, which
is being retired on May 12, 2026). One of two image-generation models used
for the Q3 comparison.

Returns PIL.Image in RGB mode for downstream compImage / evaluation / display.
Reads OPENAI_API_KEY from .env.

Note: The gpt-image family ALWAYS returns base64-encoded images (unlike
DALL-E 3, which returned URLs by default). This wrapper decodes b64_json
to bytes and opens as PIL.

Usage and CLI Flags:
    from gen_image_gpt import generate_gpt_image
    img = generate_gpt_image(prompt)
    img.save('out.png')

Parameters:
- size    : '1024x1024' (square), '1024x1536' (portrait), '1536x1024' (landscape)
- quality : 'low' / 'medium' / 'high'. Medium is comparable-priced to
            DALL-E 3 standard (~$0.04/image). High is substantially more.
'''

import base64
import io
import os
import sys
import time

from PIL import Image
# Use for locally .env defined API keys/tokens.
from dotenv import load_dotenv


## CONSTANTS ##

GPT_IMAGE_MODEL = 'gpt-image-1.5'

# 1024x1024 matches FLUX.1-schnell's default so the two models produce
# same-resolution outputs for fair comparison.
DEFAULT_SIZE = '1024x1024'        # also '1024x1536', '1536x1024'
DEFAULT_QUALITY = 'medium'        # 'low' | 'medium' | 'high'
DEFAULT_OUTPUT_FORMAT = 'png'     # 'png' | 'jpeg'


## HELPERS ##

def _client():
    '''Return an OpenAI client using OPENAI_API_KEY from .env.'''
    load_dotenv()
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not found in .env or environment.')
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            '`openai` package not installed. Run: pip install openai python-dotenv'
        )
    return OpenAI(api_key=api_key)


## PUBLIC API ##

def generate_gpt_image(
    prompt: str,
    size: str = DEFAULT_SIZE,
    quality: str = DEFAULT_QUALITY,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> Image.Image:
    '''Generate one image via gpt-image-1.5. Returns PIL.Image in RGB mode.
    Raises on API error; OpenAI SDK handles transient retries internally.'''
    client = _client()
    resp = client.images.generate(
        model=GPT_IMAGE_MODEL,
        prompt=prompt,
        size=size,
        quality=quality,
        output_format=output_format,
        n=1,
    )
    b64 = resp.data[0].b64_json
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')


## SMOKE TEST ##

if __name__ == '__main__':
    prompt = (
        sys.argv[1] if len(sys.argv) > 1
        else 'a wooden chess set on a dark mahogany table, studio lighting'
    )
    print(f'Generating: {prompt!r}')
    t0 = time.time()
    img = generate_gpt_image(prompt)
    elapsed = time.time() - t0
    out_path = 'gpt_image_smoke_test.png'
    img.save(out_path)
    print(f'Saved {img.size[0]}x{img.size[1]} to {out_path} in {elapsed:.1f}s')
