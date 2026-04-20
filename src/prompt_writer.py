'''prompt_writer.py
----------------
Mistral-backed iterative prompt-refinement.

The PromptWriter class holds a current prompt string and a descriptiveness
scalar, pulls batches of reviews from a PyTorch DataLoader, and iteratively
refines the prompt via Mistral-7B-Instruct. Two public methods:
    stepPrompt(batch)  | rewrite prompt using the batch's reviews (stochastic)
    ratePrompt()       | score current prompt's descriptiveness 0-100 (deterministic)
improvementLoop wraps them both until a threshold is hit or iterMax is reached.

Mistral is loaded separately via load_mistral() and passed in, so a single
model can be reused across many PromptWriter instances (one per product run).
'''

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


## CONSTANTS ##

DEFAULT_MISTRAL_MODEL = 'mistralai/Mistral-7B-Instruct-v0.3'
DEFAULT_MAX_LENGTH = 2048
DEFAULT_PROMPT_FALLBACK = 'A high-quality product photo.'

# stepPrompt sampling params (stochastic refinement)
STEP_MAX_NEW_TOKENS = 150
STEP_TEMPERATURE = 0.7
STEP_REPETITION_PENALTY = 1.2

# ratePrompt sampling params (deterministic, terse score)
RATE_MAX_NEW_TOKENS = 10

# prompt text gets truncated to this many characters per stepPrompt
PROMPT_CHAR_CAP = 500


## MISTRAL LOADER ##

def load_mistral(model_name: str = DEFAULT_MISTRAL_MODEL):
    '''Load Mistral-7B-Instruct with 4-bit quantization via bitsandbytes.
    Returns (model, tokenizer). Requires CUDA and gated-model access via
    HUGGINGFACE_TOKEN (set in .env or via `huggingface-cli login`).'''
    print(f'Loading {model_name} (4-bit quantization)...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='cuda',
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
        ),
    )
    model.config.max_length = DEFAULT_MAX_LENGTH
    print('Mistral model loaded successfully.')
    return model, tokenizer


## PROMPT WRITER ##

# System prompts are frozen content. Changing them changes stepPrompt /
# ratePrompt behavior, Keep them verbatim unless you're intentionally
# iterating the orchestration prompts.
STEP_PROMPT_SYSTEM = (
    'You are a professional image prompt engineer. Rewrite the prompt to be '
    'more descriptive based on the reviews. Output ONLY a single paragraph '
    'of descriptive English. Be concise and focus on visual features. No '
    'labels, no titles.'
)

RATE_PROMPT_SYSTEM = (
    'You are a hyper-critical visual quality rater. Rate the prompt from '
    '0.0 to 100.0. \n\nSTRICT SCORING RULES:\n'
    '- 0-30: Generic, vague, or contains fluff/labels.\n'
    '- 31-60: Descriptive but common; lacks specific lighting, texture, or composition details.\n'
    '- 61-85: Very good, detailed visual descriptions. Most good prompts fall here.\n'
    '- 86-100: Absolute perfection. Must be incredibly precise, concise, and visually evocative. \n\n'
    "Do NOT give high scores easily. If it is just 'fine', give it a 40. "
    'Only output the float number.'
)


class PromptWriter:
    '''Iterative prompt refiner backed by Mistral-7B.

    Args:
        dataloader: PyTorch DataLoader over review batches
            (list[dict] per batch, each dict has 'title' and 'text').
        model, tokenizer: loaded via load_mistral().
        initial_prompt_path: optional path to a seed initial_prompt.txt.
            If None or missing, falls back to DEFAULT_PROMPT_FALLBACK.
    '''

    def __init__(self, dataloader, model, tokenizer, initial_prompt_path: str | None = None):
        self.dataloader = dataloader
        self.descriptiveness = 0.0
        self.model = model
        self.tokenizer = tokenizer

        self.prompt = self._load_initial_prompt(initial_prompt_path)

        # HF pipeline is expensive to create. Reuse per-PromptWriter.
        self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)

    @staticmethod
    def _load_initial_prompt(path: str | None) -> str:
        if path is None:
            return DEFAULT_PROMPT_FALLBACK
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return text or DEFAULT_PROMPT_FALLBACK
        except Exception:
            return DEFAULT_PROMPT_FALLBACK

    def stepPrompt(self, batch):
        '''Rewrite self.prompt using the batch's reviews (stochastic sampling).'''
        reviews_summary = '\n'.join(
            f"- {r.get('title', '')}: {r.get('text', '')[:120]}" for r in batch
        )
        messages = [
            {'role': 'system', 'content': STEP_PROMPT_SYSTEM},
            {'role': 'user', 'content':
                f'Current Prompt: {self.prompt}\n\nNew details to add: {reviews_summary}'},
        ]

        output = self.generator(
            messages,
            max_new_tokens=STEP_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=STEP_TEMPERATURE,
            repetition_penalty=STEP_REPETITION_PENALTY,
            use_cache=False,
        )

        raw_text = output[0]['generated_text'][-1]['content'].strip()
        # Strip non-ASCII (the pipeline occasionally emits unicode artifacts)
        # and collapse whitespace; truncate to PROMPT_CHAR_CAP.
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', raw_text)
        self.prompt = ' '.join(cleaned_text.split())[:PROMPT_CHAR_CAP]

    def ratePrompt(self):
        '''Score self.prompt on descriptiveness 0-100 (deterministic).'''
        messages = [
            {'role': 'system', 'content': RATE_PROMPT_SYSTEM},
            {'role': 'user', 'content':
                f'Rate this image prompt for technical visual precision: {self.prompt}'},
        ]

        output = self.generator(
            messages,
            max_new_tokens=RATE_MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False,
        )

        raw_score = output[0]['generated_text'][-1]['content'].strip()
        try:
            score_match = re.search(r'([0-9]*\.?[0-9]+)', raw_score)
            self.descriptiveness = float(score_match.group(1)) if score_match else 0.0
        except (ValueError, AttributeError):
            self.descriptiveness = 0.0

        # Clamp to the documented 0-100 range.
        self.descriptiveness = max(0.0, min(100.0, self.descriptiveness))

    def improvementLoop(self, descriptivenessThreshold: float, iterMax: int) -> int:
        '''Iterate stepPrompt + ratePrompt until descriptiveness >= threshold
        OR iterMax reached OR dataloader exhausted. Returns iterations taken.'''
        iters = 0
        data_iter = iter(self.dataloader)
        while self.descriptiveness < descriptivenessThreshold and iters < iterMax:
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            self.stepPrompt(batch)
            self.ratePrompt()
            iters += 1
        return iters
