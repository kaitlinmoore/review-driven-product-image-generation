'''reviews_dataloader.py
---------------------
PyTorch Dataset and DataLoader adapter for reviews_ranked.jsonl.

Plugs into PromptWriter(dataloader=...) so stepPrompt() can
pull batches of review data during the improvement loop.

Usage:
    from reviews_dataloader import make_reviews_dataloader
    dl = make_reviews_dataloader('chess_set')
    pw = PromptWriter(dl)
    pw.prompt = open('data/chess_set/initial_prompt.txt').read()
    pw.improvementLoop(threshold, iterMax)

'''

import json
import os

from torch.utils.data import Dataset, DataLoader

## CONSTANTS ##

DEFAULT_BATCH_SIZE = 15

# DEFAULT_SKIP_TOP must match DEFAULT_TOP_N in build_prompt_context.py so the initial prompt's
# reviews and the dataloader's reviews don't overlap or leave a gap. Update
# both together if you change this.
DEFAULT_SKIP_TOP = 30
DEFAULT_DATA_DIR = 'data'

REVIEW_FIELDS = (
    'rating',
    'title',
    'text',
    'helpful_vote',
    'timestamp',
    '_rank',
    '_gate_reason',
    '_text_length',
)


class ReviewsDataset(Dataset):
    def __init__(self, path: str, skip_top: int = DEFAULT_SKIP_TOP):
        if not os.path.exists(path):
            raise FileNotFoundError(f'reviews file not found: {path}')
        with open(path, 'r', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if skip_top < 0:
            raise ValueError(f'skip_top must be >= 0, got {skip_top}')
        self.path = path
        self.skip_top = skip_top
        self.rows = rows[skip_top:]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {k: r.get(k) for k in REVIEW_FIELDS}


def make_reviews_dataloader(
    product_slug: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    skip_top: int = DEFAULT_SKIP_TOP,
    data_dir: str = DEFAULT_DATA_DIR,
) -> DataLoader:
    '''Return a DataLoader over reviews_ranked.jsonl for the given product.
    The batch each iteration yields is a `list[dict]` of review records.'''
    path = os.path.join(data_dir, product_slug, 'reviews_ranked.jsonl')
    dataset = ReviewsDataset(path, skip_top=skip_top)
    # collate_fn=list keeps each batch as a plain list of dicts — avoids
    # PyTorch stacking numeric fields into tensors while leaving strings
    # as separate lists, which would be awkward for prompt assembly.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=list,
    )


if __name__ == '__main__':
    import sys
    slug = sys.argv[1] if len(sys.argv) > 1 else 'chess_set'
    dl = make_reviews_dataloader(slug)
    ds = dl.dataset
    print(f'Product: {slug}')
    print(f'  skip_top: {ds.skip_top}')
    print(f'  available reviews: {len(ds):,}')
    print(f'  batch_size: {dl.batch_size}')
    print(f'  total batches: {len(dl):,}')
    batch = next(iter(dl))
    print(f'\nFirst batch: list of {len(batch)} dicts')
    print(f'Fields per item: {list(batch[0].keys())}')
    print(f'\nFirst item preview:')
    first = batch[0]
    for k, v in first.items():
        if isinstance(v, str) and len(v) > 100:
            v = v[:100] + '...'
        print(f'  {k}: {v!r}')
