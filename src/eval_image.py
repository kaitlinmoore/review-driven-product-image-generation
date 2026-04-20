'''eval_image.py
-------------
Image similarity metrics for the generated-vs-ground-truth comparison stage.

Two comparison modes are supported:

1. Image-vs-text (via CLIP)
   - compImage(image, ground_truth_text) -> float
     Used in the canonical agentLoop flow: compare the generated PIL image to
     the ground-truth product TITLE string.

2. Image-vs-image (via CLIP / DINOv2 / SigLIP)
   - clip_image_similarity(img_cand, img_gt) -> float
   - dinov2_similarity(img_cand, img_gt) -> float
   - siglip_similarity(img_cand, img_gt) -> float

Models are loaded lazily and cached module-wide so the first call to each
similarity function pays a one-time download/load cost. Subsequent calls are
cheap.
'''

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel, CLIPProcessor


## MODEL IDS ##

ST_CLIP_NAME = 'clip-ViT-B-32'                         # sentence-transformers CLIP (image+text)
HF_CLIP_NAME = 'openai/clip-vit-base-patch32'          # transformers CLIP (image features)
DINOV2_NAME = 'facebook/dinov2-base'
SIGLIP_NAME = 'google/siglip-base-patch16-224'


## LAZY MODEL CACHES ##

_ST_CLIP = None          # sentence-transformers SentenceTransformer
_HF_CLIP = None          # tuple (processor, model) for transformers CLIP
_DINO = None             # tuple (processor, model)
_SIGLIP = None           # tuple (processor, model)


def _device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _get_st_clip() -> SentenceTransformer:
    '''Lazy-load the sentence-transformers CLIP for image-vs-text comparisons.'''
    global _ST_CLIP
    if _ST_CLIP is None:
        _ST_CLIP = SentenceTransformer(ST_CLIP_NAME)
    return _ST_CLIP


def _get_hf_clip():
    '''Lazy-load transformers CLIP (returns processor, model) for image-vs-image.'''
    global _HF_CLIP
    if _HF_CLIP is None:
        processor = CLIPProcessor.from_pretrained(HF_CLIP_NAME)
        model = CLIPModel.from_pretrained(HF_CLIP_NAME).to(_device()).eval()
        _HF_CLIP = (processor, model)
    return _HF_CLIP


def _get_dino():
    '''Lazy-load DINOv2 (returns processor, model).'''
    global _DINO
    if _DINO is None:
        processor = AutoImageProcessor.from_pretrained(DINOV2_NAME)
        model = AutoModel.from_pretrained(DINOV2_NAME).to(_device()).eval()
        _DINO = (processor, model)
    return _DINO


def _get_siglip():
    '''Lazy-load SigLIP (returns processor, model).'''
    global _SIGLIP
    if _SIGLIP is None:
        processor = AutoProcessor.from_pretrained(SIGLIP_NAME)
        model = AutoModel.from_pretrained(SIGLIP_NAME).to(_device()).eval()
        _SIGLIP = (processor, model)
    return _SIGLIP


## IMAGE-VS-TEXT (CLIP) | canonical flow ##

def compImage(image_input, ground_truth_text: str) -> float:
    '''CLIP cosine similarity between a generated image and a ground-truth
    text description (e.g. the product title).

    image_input: either a PIL.Image or a file path (str/Path).
    ground_truth_text: the string to compare against.

    Returns a float in [-1, 1] (practically almost always in [0, 1]).'''
    model = _get_st_clip()

    if isinstance(image_input, (str, Path)):
        img_p = Path(image_input)
        if not img_p.exists():
            raise FileNotFoundError(f'Candidate image not found: {img_p}')
        img_cand = Image.open(img_p).convert('RGB')
    else:
        img_cand = image_input.convert('RGB')

    img_emb = model.encode(img_cand, convert_to_tensor=True)
    txt_emb = model.encode(ground_truth_text, convert_to_tensor=True)
    return util.cos_sim(img_emb, txt_emb).item()


def evalImage(generated_image, ground_truth_text: str) -> float:
    '''Thin wrapper over compImage so the orchestration loop can import a
    single `evalImage` symbol. Kept for interface compatibility with the
    notebook's agentLoop.'''
    return compImage(generated_image, ground_truth_text)


## IMAGE-VS-IMAGE (CLIP / DINOv2 / SigLIP) ##
# These three functions mirror the REPRODUCIBILITY.md pinned-models list.

def _cosine_similarity_from_features(features: torch.Tensor) -> float:
    '''Cosine similarity between rows 0 and 1 of a (2, D) feature tensor.'''
    return F.cosine_similarity(features[0:1], features[1:2]).item()


def clip_image_similarity(img_cand_pil: Image.Image, img_gt_pil: Image.Image) -> float:
    '''Image-vs-image cosine similarity using transformers CLIP image features.'''
    processor, model = _get_hf_clip()
    inputs = processor(images=[img_cand_pil, img_gt_pil], return_tensors='pt').to(_device())
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)  # (2, 512)
    return _cosine_similarity_from_features(embeddings)


def dinov2_similarity(img_cand_pil: Image.Image, img_gt_pil: Image.Image) -> float:
    '''Image-vs-image cosine similarity using DINOv2 CLS-token embeddings.'''
    processor, model = _get_dino()
    inputs = processor(images=[img_cand_pil, img_gt_pil], return_tensors='pt').to(_device())
    with torch.no_grad():
        outputs = model(**inputs)
    # CLS token | Shape (2, hidden_dim)
    embeddings = outputs.last_hidden_state[:, 0]
    return _cosine_similarity_from_features(embeddings)


def siglip_similarity(img_cand_pil: Image.Image, img_gt_pil: Image.Image) -> float:
    '''Image-vs-image cosine similarity using SigLIP image features.
    (Bug fix from cell 1's pseudocode: original reference to `processor` and
    `model` didn't call _get_siglip() to bind them.)'''
    processor, model = _get_siglip()
    inputs = processor(images=[img_cand_pil, img_gt_pil], return_tensors='pt').to(_device())
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)  # (2, 768)
    return _cosine_similarity_from_features(embeddings)
